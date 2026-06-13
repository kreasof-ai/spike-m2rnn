"""
STAGE 0 -- a SPIKING, matrix-valued-state RNN (a spiking take on M2RNN) trained
with EGGROLL evolution strategies on char-level Shakespeare.

This is the CORRECTNESS REFERENCE, not the fast path. Read this before running:

  * FLOAT weights + EGGROLL low-rank forward. NO ternary/BitNet quantization yet,
    NO bit-packed kernel, NO on-the-fly noise. The membrane is a float accumulator
    and the leak is a real multiply. That is on purpose: this stage answers ONE
    question -- does a spiking nonlinear matrix-RNN produce a learning signal under
    ES at all? Quantization is Stage 1 (and it breaks the low-rank "never
    materialise" trick -> per-member 2-bit weights). The Triton popcount +
    in-SRAM Philox kernel is Stage 2. Add them only after this learns.

  * MODE = "spike" is the architecture you want to test. MODE = "tanh" recovers
    analog M2RNN (eqs 10-12: Z=tanh(HW + k vᵀ); H=fH+(1-f)Z; y=Hᵀq) as a baseline,
    so a failure tells you whether it's the spiking nonlinearity or ES itself.

  * The q/k/v/forget projections are computed in PARALLEL across the sequence
    (M2RNN's point); only the recurrence is sequential. Parallelism for the GPU
    lives in the population (P) and batch (B) axes, not the sequence.

  * DEAD-ZONE WARNING (this is the ternary bin-width problem arriving early): a
    hard spike threshold makes the loss piecewise-CONSTANT in every weight that
    feeds a spike -- only the head sees a smooth landscape. A weight perturbation
    that flips no spikes yields zero fitness signal. That's why SIGMA here is 0.05,
    not the ViT's 0.01. If the loss is flat, the first knob is SIGMA (then POP,
    then THRESHOLD/DECAY to get firing into a healthy ~5-30% band).

What Shakespeare tests: that the full ES + spiking + recurrence pipeline learns
end-to-end (val loss falls, samples acquire character statistics). It does NOT
test the state-tracking thesis that motivates the architecture -- for that you
need S3/S5 permutation tasks with length generalisation. This is the smoke test
that earns the right to run those.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================ config ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16          # fp32 is safest for the threshold dynamics; try bf16 on GPU
MODE   = "spike"                # "spike" (your idea) | "tanh" (analog M2RNN baseline)

# model -- deliberately SMALL. ES wants few params + large population.
DIM     = 128
DEPTH   = 2
K_DIM   = 64                    # key head dim
V_DIM   = 64                    # value head dim   (matrix state is K_DIM x V_DIM)
MLP_DIM = 256
BLOCK   = 64                    # context length (kept short: the recurrence is sequential)

# spiking dynamics (fixed for stage 0; make these learnable later)
THRESHOLD = 1.0
DECAY     = 0.9                 # ~7-step half-life: fine for char-LM, NOT for long-range state-tracking

# EGGROLL / ES
BATCH_SIZE = 24
POP_SIZE   = 512                # raise as far as memory allows; chunk if OOM
RANK       = 1
SIGMA      = 0.05               # larger than the ViT's 0.01 ON PURPOSE -- see DEAD-ZONE warning above
LR         = 0.05
CHUNK      = None               # e.g. 128 to slice the population; None = all at once
RANK_SCALE = 1.0 / math.sqrt(RANK)

torch.set_float32_matmul_precision("high")


# ===================== memory-efficient EGGROLL ops =====================
# (identical in spirit to your ViT: shared base GEMM + cheap low-rank correction)
def eggroll_linear(x, weight, A, B, bias=None, bias_noise=None, sigma=SIGMA, rank_scale=RANK_SCALE):
    base = F.linear(x, weight)
    if x.dim() == 3:                                  # (B,S,I): broadcast a new pop dim
        lr = torch.einsum("bsi,pir->bspr", x, B)
        lr = torch.einsum("bspr,por->pbso", lr, A)
        out = base.unsqueeze(0) + (sigma * rank_scale) * lr
    else:                                             # (P,B,S,I): pop dim already there
        lr = torch.einsum("pbsi,pir->pbsr", x, B)
        lr = torch.einsum("pbsr,por->pbso", lr, A)
        out = base + (sigma * rank_scale) * lr
    if bias is not None:
        out = out + bias
    if bias_noise is not None:
        out = out + sigma * bias_noise[:, None, None, :]
    return out


def eggroll_ln(x, g_base, g_noise, b_base, b_noise, sigma=SIGMA, eps=1e-5):
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    xn  = (x - mu) * torch.rsqrt(var + eps)
    g   = g_base + sigma * g_noise
    b   = b_base + sigma * b_noise
    return xn * g[:, None, None, :] + b[:, None, None, :]


# ===================== the model (base params only) =====================
class SpikingM2RNN(nn.Module):
    def __init__(self, vocab, dim=DIM, depth=DEPTH, k=K_DIM, v=V_DIM, mlp=MLP_DIM, mode=MODE):
        super().__init__()
        self.dim, self.depth, self.k, self.v, self.mode, self.vocab = dim, depth, k, v, mode, vocab

        Pd = nn.ParameterDict()
        def mat(name, o, i):
            Pd[name + "_w"] = nn.Parameter(torch.empty(o, i))
            Pd[name + "_b"] = nn.Parameter(torch.zeros(o))
        def lnp(name, d):
            Pd[name + "_g"] = nn.Parameter(torch.ones(d))
            Pd[name + "_b"] = nn.Parameter(torch.zeros(d))

        mat("embed", dim, vocab)                       # token embed == linear on one-hot (NoPE: order from recurrence)
        for L in range(depth):
            p = f"block{L}"
            lnp(p + "_ln1", dim)
            mat(p + "_q", k, dim)
            mat(p + "_k", k, dim)
            mat(p + "_v", v, dim)
            if mode == "tanh":
                mat(p + "_f", 1, dim)                  # forget-gate logits (analog M2RNN only)
            mat(p + "_W", v, v)                        # state transition (V x V) -- this is the BitNet-able matrix
            mat(p + "_o", dim, v)                      # readout projection V -> dim
            lnp(p + "_ln2", dim)
            mat(p + "_fc1", mlp, dim)
            mat(p + "_fc2", dim, mlp)
        lnp("norm", dim)
        mat("head", vocab, dim)
        self.P = Pd

        for name, par in Pd.items():
            if par.dim() == 2:
                nn.init.kaiming_uniform_(par, a=math.sqrt(5))

    # ---- per-step noise: matrices -> (A,B) low-rank factors; everything else dense ----
    @torch.no_grad()
    def sample_noise(self, pop, rank, device, dtype):
        noise = {}
        for name, par in self.P.items():
            if name.endswith("_w"):
                o, i = par.shape
                noise[name] = (torch.randn(pop, o, rank, device=device, dtype=dtype),
                               torch.randn(pop, i, rank, device=device, dtype=dtype))
            else:
                noise[name] = torch.randn(pop, *par.shape, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def zero_noise(self, device, dtype):
        noise = {}
        for name, par in self.P.items():
            if name.endswith("_w"):
                o, i = par.shape
                noise[name] = (torch.zeros(1, o, RANK, device=device, dtype=dtype),
                               torch.zeros(1, i, RANK, device=device, dtype=dtype))
            else:
                noise[name] = torch.zeros(1, *par.shape, device=device, dtype=dtype)
        return noise

    def _lin(self, x, name, noise, sigma):
        A, B = noise[name + "_w"]
        return eggroll_linear(x, self.P[name + "_w"], A, B,
                              bias=self.P[name + "_b"], bias_noise=noise[name + "_b"], sigma=sigma)

    def _ln(self, x, name, noise, sigma):
        return eggroll_ln(x, self.P[name + "_g"], noise[name + "_g"],
                          self.P[name + "_b"], noise[name + "_b"], sigma=sigma)

    def forward(self, idx, noise, sigma=SIGMA, return_fire=False):
        # idx: (B,T) long
        Pn = noise["head_w"][0].shape[0]
        Bn, Tn = idx.shape
        k_, v_ = self.k, self.v
        oh = F.one_hot(idx, num_classes=self.vocab).to(self.P["embed_w"].dtype)   # (B,T,vocab)
        x = self._lin(oh, "embed", noise, sigma)                                  # (P,B,T,dim)

        fire_acc, fire_n = 0.0, 0
        for L in range(self.depth):
            p = f"block{L}"
            xn = self._ln(x, p + "_ln1", noise, sigma)
            Q  = self._lin(xn, p + "_q", noise, sigma)        # (P,B,T,k)
            Kp = self._lin(xn, p + "_k", noise, sigma)        # (P,B,T,k)
            Vp = self._lin(xn, p + "_v", noise, sigma)        # (P,B,T,v)
            if self.mode == "spike":
                Q  = (Q  > 0).to(x.dtype)                     # spike-encode q/k/v (threshold at 0; bias = -threshold)
                Kp = (Kp > 0).to(x.dtype)
                Vp = (Vp > 0).to(x.dtype)
                f_t = None
            else:                                             # analog M2RNN
                f_t = torch.sigmoid(self._lin(xn, p + "_f", noise, sigma))   # (P,B,T,1)

            state = torch.zeros(Pn, Bn, k_, v_, device=x.device, dtype=x.dtype)   # H (analog) | S_prev (spiking)
            mem   = torch.zeros(Pn, Bn, k_, v_, device=x.device, dtype=x.dtype)   # U (spiking only)
            ys = []
            for t in range(Tn):
                kt = Kp[:, :, t, :]; vt = Vp[:, :, t, :]; qt = Q[:, :, t, :]
                outer = torch.einsum("pbk,pbv->pbkv", kt, vt)                      # k_t ⊗ v_t
                trans = self._lin(state, p + "_W", noise, sigma)                   # transition on PREVIOUS state
                if self.mode == "spike":
                    mem = DECAY * mem + trans + outer
                    S   = (mem > THRESHOLD).to(x.dtype)
                    mem = mem * (1.0 - S)                                          # hard reset
                    yt  = torch.einsum("pbkv,pbk->pbv", S, qt)
                    state = S
                    if return_fire:
                        fire_acc += S.mean().item(); fire_n += 1
                else:
                    Z  = torch.tanh(trans + outer)
                    ft = f_t[:, :, t, :].unsqueeze(-1)                             # (P,B,1,1)
                    state = ft * state + (1.0 - ft) * Z
                    yt = torch.einsum("pbkv,pbk->pbv", state, qt)
                ys.append(yt)
            Y = torch.stack(ys, dim=2)                                            # (P,B,T,v)
            x = x + self._lin(Y, p + "_o", noise, sigma)                          # residual

            xn2 = self._ln(x, p + "_ln2", noise, sigma)
            h   = F.gelu(self._lin(xn2, p + "_fc1", noise, sigma))
            x   = x + self._lin(h, p + "_fc2", noise, sigma)

        x = self._ln(x, "norm", noise, sigma)
        logits = self._lin(x, "head", noise, sigma)                               # (P,B,T,vocab)
        if return_fire:
            return logits, (fire_acc / max(fire_n, 1))
        return logits


# ===================== ES loss / fitness =====================
def per_member_loss(logits, targets):
    P, B, T, C = logits.shape
    flat = logits.reshape(P * B * T, C).float()
    tgt  = targets.reshape(B * T).repeat(P)            # tile (B*T) block P times -> matches p-major flatten
    return F.cross_entropy(flat, tgt, reduction="none").reshape(P, B * T).mean(1)   # (P,)


def fitness_from_loss(loss):
    ranks = torch.argsort(torch.argsort(loss, descending=True))
    return ranks.float() / (loss.shape[0] - 1) - 0.5


# ===================== data (char-level, nanoGPT convention) =====================
def load_data(path="input.txt"):
    text = open(path, "r").read()
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for i, c in enumerate(chars)}
    ids   = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n     = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars), stoi, itos


def get_batch(data, batch, block, device):
    ix = torch.randint(len(data) - block - 1, (batch,)).tolist()
    x  = torch.stack([data[i:i + block]       for i in ix])
    y  = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)


# ===================== train / eval / sample =====================
@torch.no_grad()
def evaluate(model, va, vocab, step, train_loss):
    zn = model.zero_noise(DEVICE, DTYPE)
    losses, fire = [], None
    for _ in range(20):
        x, y = get_batch(va, BATCH_SIZE, BLOCK, DEVICE)
        out = model(x, zn, 0.0, return_fire=(fire is None and model.mode == "spike"))
        logits, fire = out if isinstance(out, tuple) else (out, fire)
        losses.append(F.cross_entropy(logits[0].reshape(-1, vocab).float(), y.reshape(-1)).item())
    vloss = sum(losses) / len(losses)
    extra = f" | fire {fire:.3f}" if fire is not None else ""
    print(f"step {step:05d} | train(min) {train_loss.min().item():.3f} "
          f"| val {vloss:.3f} ({vloss / math.log(2):.2f} bpc){extra}")


@torch.no_grad()
def generate(model, stoi, itos, prompt="\n", n=400):
    zn  = model.zero_noise(DEVICE, DTYPE)
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=DEVICE)
    for _ in range(n):
        logits = model(idx[:, -BLOCK:], zn, 0.0)                 # (1,1,T,vocab)
        probs  = F.softmax(logits[0, :, -1, :].float(), dim=-1)  # (1,vocab)
        idx    = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())


def train(steps=3000, eval_every=100, data_path="input.txt"):
    tr, va, vocab, stoi, itos = load_data(data_path)
    model = SpikingM2RNN(vocab).to(DEVICE).to(DTYPE)
    model.eval(); model.requires_grad_(False)
    model = torch.compile(model)
    coeff   = LR / (POP_SIZE * SIGMA)
    nparams = sum(p.numel() for p in model.P.values())
    print(f"mode={MODE} params={nparams:,} pop={POP_SIZE} sigma={SIGMA} "
          f"block={BLOCK} dim={DIM} K={K_DIM} V={V_DIM} device={DEVICE}")

    for step in range(1, steps + 1):
        x, y  = get_batch(tr, BATCH_SIZE, BLOCK, DEVICE)
        noise = model.sample_noise(POP_SIZE, RANK, DEVICE, DTYPE)
        with torch.no_grad():
            if CHUNK is None:
                loss = per_member_loss(model(x, noise, SIGMA), y)
            else:
                parts = []
                for s in range(0, POP_SIZE, CHUNK):
                    sub = {k: (tuple(t[s:s + CHUNK] for t in vv) if isinstance(vv, tuple) else vv[s:s + CHUNK])
                           for k, vv in noise.items()}
                    parts.append(per_member_loss(model(x, sub, SIGMA), y))
                loss = torch.cat(parts)
        fit = fitness_from_loss(loss).to(DTYPE)

        for name, par in model.P.items():
            if name.endswith("_w"):
                A, B = noise[name]
                upd = RANK_SCALE * torch.einsum("p,por,pir->oi", fit, A, B)
            else:
                upd = torch.tensordot(fit, noise[name], dims=([0], [0]))
            par.data.add_(coeff * upd.to(par.dtype))

        if step == 1 or step % eval_every == 0:
            evaluate(model, va, vocab, step, loss)

    print("\n--- sample ---")
    print(generate(model, stoi, itos))


if __name__ == "__main__":
    train()