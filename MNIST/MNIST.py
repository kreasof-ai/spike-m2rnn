"""
EGGROLL ViT with a memory-efficient low-rank forward pass.

Key difference vs. the naive vmap implementation:
  - We NEVER materialise a perturbed weight (M + sigma*E) per population member.
  - For every matrix M we keep ONE shared base weight + per-member factors A,B.
  - The forward computes the shared base matmul once (high arithmetic intensity)
    and adds the cheap low-rank correction (sigma/sqrt(r)) * (x @ B) @ A^T.

This removes the O(P * m * n) weight/noise blow-up. Activations are still
O(P * B * S * D) (and scores are O(P * B * heads * S^2)), so chunk over the
population (see CHUNK) if that is the bottleneck.

Core forward/update math is numerically equivalent to brute-force weight
materialisation (verified to ~1e-15 in float64).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Hyperparameters
# ==========================================
BATCH_SIZE = 64
POP_SIZE   = 4096         # population (the paper pushes this far higher; chunk if OOM)
RANK       = 1
SIGMA      = 0.01
LR         = 0.05
CHUNK      = 1024          # e.g. 1024 to process the population in slices; None = all at once
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE      = torch.float16 # bfloat16 recommended on Ampere+; fp16 softmax/CE can be touchy

RANK_SCALE = 1.0 / math.sqrt(RANK)
torch.set_float32_matmul_precision("high")


# ==========================================
# 2. Memory-efficient EGGROLL linear op
#    x:      (B,S,I)   -> (P,B,S,O)   [first layer, no pop dim in]
#       or   (P,B,S,I) -> (P,B,S,O)   [later layers, pop dim already present]
#    A:(P,O,r)  B:(P,I,r)  bias:(O,) shared  bias_noise:(P,O) raw N(0,1)
# ==========================================
def eggroll_linear(x, weight, A, B, bias=None, bias_noise=None,
                   sigma=SIGMA, rank_scale=RANK_SCALE):
    base = F.linear(x, weight)                       # shared base; one big GEMM either way
    if x.dim() == 3:                                 # (B,S,I): broadcast a new pop dim
        lr = torch.einsum("bsi,pir->bspr", x, B)
        lr = torch.einsum("bspr,por->pbso", lr, A)
        out = base.unsqueeze(0) + (sigma * rank_scale) * lr
    else:                                            # (P,B,S,I): pop dim already there
        lr = torch.einsum("pbsi,pir->pbsr", x, B)
        lr = torch.einsum("pbsr,por->pbso", lr, A)
        out = base + (sigma * rank_scale) * lr
    if bias is not None:
        out = out + bias                             # broadcast over (P,B,S)
    if bias_noise is not None:
        out = out + sigma * bias_noise[:, None, None, :]
    return out                                       # (P,B,S,O)


def eggroll_ln(x, g_base, g_noise, b_base, b_noise, sigma=SIGMA, eps=1e-5):
    # x:(P,B,S,D). gains/biases are perturbed too: g,b become (P,D).
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    xn  = (x - mu) * torch.rsqrt(var + eps)
    g   = g_base + sigma * g_noise                   # (P,D)
    b   = b_base + sigma * b_noise
    return xn * g[:, None, None, :] + b[:, None, None, :]


# ==========================================
# 3. ViT holding only BASE parameters (no replication).
#    Per-step noise is a dict: matrix entries are (A,B) tuples; everything else
#    is a raw N(0,1) tensor of shape (P, *param.shape). Keys use '_' (ParameterDict
#    forbids '.').
# ==========================================
class BatchedViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10,
                 dim=64, depth=2, heads=4, mlp=128):
        super().__init__()
        assert image_size % patch_size == 0
        self.dim, self.depth, self.heads = dim, depth, heads
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2                  # single channel
        self.seq = n_patches + 1                      # + cls token

        Pd = nn.ParameterDict()
        def mat(name, o, i):
            Pd[name + "_w"] = nn.Parameter(torch.empty(o, i))
            Pd[name + "_b"] = nn.Parameter(torch.zeros(o))
        def lnp(name, d):
            Pd[name + "_g"] = nn.Parameter(torch.ones(d))
            Pd[name + "_b"] = nn.Parameter(torch.zeros(d))

        mat("patch_embed", dim, patch_dim)
        Pd["cls"] = nn.Parameter(torch.randn(dim) * 0.02)
        Pd["pos"] = nn.Parameter(torch.randn(self.seq, dim) * 0.02)
        for L in range(depth):
            p = f"block{L}"
            lnp(p + "_ln1", dim)
            for proj in ("q", "k", "v", "o"):
                mat(f"{p}_{proj}", dim, dim)
            lnp(p + "_ln2", dim)
            mat(f"{p}_fc1", mlp, dim)
            mat(f"{p}_fc2", dim, mlp)
        lnp("norm", dim)
        mat("head", num_classes, dim)
        self.P = Pd

        for name, par in Pd.items():
            if par.dim() == 2:
                nn.init.kaiming_uniform_(par, a=math.sqrt(5))

    @torch.no_grad()
    def sample_noise(self, pop, rank, device, dtype):
        noise = {}
        for name, par in self.P.items():
            if name.endswith("_w"):                     # linear weight -> low-rank factors
                o, i = par.shape
                noise[name] = (torch.randn(pop, o, rank, device=device, dtype=dtype),
                               torch.randn(pop, i, rank, device=device, dtype=dtype))
            else:                                       # bias / LN gain / cls / pos -> dense
                noise[name] = torch.randn(pop, *par.shape, device=device, dtype=dtype)
        return noise

    def forward(self, x_img, noise, sigma=SIGMA):
        P = noise["head_w"][0].shape[0]
        D, h = self.dim, self.heads

        def lin(x, name):
            A, B = noise[name + "_w"]
            return eggroll_linear(x, self.P[name + "_w"], A, B,
                                  bias=self.P[name + "_b"],
                                  bias_noise=noise[name + "_b"], sigma=sigma)

        def ln(x, name):
            return eggroll_ln(x, self.P[name + "_g"], noise[name + "_g"],
                              self.P[name + "_b"], noise[name + "_b"], sigma=sigma)

        # patchify (B,1,28,28) -> (B, n_patches, patch_dim)
        B = x_img.shape[0]
        ps = self.patch_size
        xp = x_img.unfold(2, ps, ps).unfold(3, ps, ps).reshape(B, 1, -1, ps, ps).squeeze(1)
        xp = xp.reshape(B, -1, ps * ps)

        x = lin(xp, "patch_embed")                    # (P,B,n_patches,D) <- pop dim born here
        cls = (self.P["cls"] + sigma * noise["cls"])[:, None, None, :].expand(P, B, 1, D)
        x   = torch.cat([cls, x], dim=2)              # (P,B,S,D)
        pos = (self.P["pos"] + sigma * noise["pos"])  # (P,S,D)
        x   = x + pos[:, None, :, :]

        for L in range(self.depth):
            p = f"block{L}"
            y = ln(x, p + "_ln1")                      # attention sublayer (pre-norm)
            Pn, Bn, Sn, _ = y.shape
            q = lin(y, p + "_q").reshape(Pn, Bn, Sn, h, D // h).transpose(2, 3)  # (P,B,h,S,dh)
            k = lin(y, p + "_k").reshape(Pn, Bn, Sn, h, D // h).transpose(2, 3)
            v = lin(y, p + "_v").reshape(Pn, Bn, Sn, h, D // h).transpose(2, 3)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(2, 3).reshape(Pn, Bn, Sn, D)
            x = x + lin(a, p + "_o")
            y = ln(x, p + "_ln2")                      # MLP sublayer
            y = F.gelu(lin(y, p + "_fc1"))
            x = x + lin(y, p + "_fc2")

        x = ln(x, "norm")
        cls_out = x[:, :, 0:1, :]                      # keep length-1 seq dim for the linear
        return lin(cls_out, "head").squeeze(2)         # (P,B,num_classes)


def per_member_loss(logits, target):
    P, B, C = logits.shape
    flat = logits.reshape(P * B, C).float()
    tgt  = target.repeat(P)                            # pop-major: matches reshape order
    return F.cross_entropy(flat, tgt, reduction="none").reshape(P, B).mean(1)  # (P,)


def fitness_from_loss(loss):
    ranks = torch.argsort(torch.argsort(loss, descending=True))
    return ranks.float() / (loss.shape[0] - 1) - 0.5


def zero_noise(model, device, dtype):
    """Noise that makes the forward use the base/mean parameters (population of 1)."""
    en = {}
    for name, par in model.P.items():
        if name.endswith("_w"):
            o, i = par.shape
            en[name] = (torch.zeros(1, o, RANK, device=device, dtype=dtype),
                        torch.zeros(1, i, RANK, device=device, dtype=dtype))
        else:
            en[name] = torch.zeros(1, *par.shape, device=device, dtype=dtype)
    return en


# ==========================================
# 4. Training loop (no vmap, no weight replication)
# ==========================================
def train():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    tr  = DataLoader(datasets.MNIST("./data", train=True,  download=True, transform=tfm),
                     batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    te  = DataLoader(datasets.MNIST("./data", train=False, download=True, transform=tfm),
                     batch_size=256, shuffle=False)

    model = BatchedViT().to(DEVICE).to(DTYPE)
    model.eval()
    model.requires_grad_(False)
    coeff = LR / (POP_SIZE * SIGMA)                    # update step size (sigma kept explicit)

    model = torch.compile(model)

    for epoch in range(1, 6):
        print(f"\n--- Epoch {epoch} ---")
        for bi, (data, target) in enumerate(tr):
            data, target = data.to(DEVICE, DTYPE), target.to(DEVICE)
            noise = model.sample_noise(POP_SIZE, RANK, DEVICE, DTYPE)

            if CHUNK is None:
                loss = per_member_loss(model(data, noise), target)
            else:
                parts = []
                for s in range(0, POP_SIZE, CHUNK):
                    sub = {k: (tuple(t[s:s+CHUNK] for t in v) if isinstance(v, tuple)
                               else v[s:s+CHUNK]) for k, v in noise.items()}
                    parts.append(per_member_loss(model(data, sub), target))
                loss = torch.cat(parts)

            fit = fitness_from_loss(loss).to(DTYPE)

            for name, par in model.P.items():
                if name.endswith("_w"):                # (1/sqrt r) sum_p f_p A_p B_p^T, no E materialised
                    A, B = noise[name]
                    upd = RANK_SCALE * torch.einsum("p,por,pir->oi", fit, A, B)
                else:
                    upd = torch.tensordot(fit, noise[name], dims=([0], [0]))
                par.data.add_(coeff * upd.to(par.dtype))

            if bi % 100 == 0:
                print(f"batch {bi:03d} | min {loss.min().item():.4f} | mean {loss.mean().item():.4f}")

        ezn = zero_noise(model, DEVICE, DTYPE)
        correct = tot = 0
        for data, target in te:
            data, target = data.to(DEVICE, DTYPE), target.to(DEVICE)
            logits = model(data, ezn)[0]               # (B,num_classes)
            correct += (logits.argmax(-1) == target).sum().item()
            tot     += target.size(0)
        print(f"--> epoch {epoch} val acc: {100*correct/tot:.2f}%")


if __name__ == "__main__":
    train()