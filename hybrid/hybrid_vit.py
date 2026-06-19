"""
Hybrid ES + GD ViT  --  a FORK of the main spiking-M2RNN project (does not import it).

Idea (partition parameter space by differentiability):

    W_eff = Q(W_latent)      # ternary {-1,0,+1}, trained GRADIENT-FREE by ES
          + U @ Vᵀ           # low-rank FP residual, trained by GRADIENT DESCENT

  * ES owns the non-differentiable part -- the ternary quantization decisions Q(.).
    ES does not care that round/clamp has zero gradient a.e.; that is its whole point.
  * GD owns the smooth part -- a small rank-r FP residual that IS differentiable, so we
    get exact gradients for that subspace instead of paying ES's O(1/sqrt(POP)) variance.

This is QLoRA *inverted*: QLoRA freezes a quantized base and trains an FP LoRA; here the
base is ALSO trained -- by the one method that can train a quantized base without a
straight-through estimator. The residual U,V are SHARED across the ES population (they are
not perturbed); only the ternary latent masters + biases/LN/cls/pos are perturbed by ES.

Consequences (both deliberate, see README):
  * FP activations throughout -- GD cannot cross a hard nonlinearity without a surrogate;
    so this fork trades multiply-free activation for GD sample-efficiency.
  * The base is materialized + quantized PER population member in the ES forward (ternary
    breaks EGGROLL's never-materialize trick); fine for a small ViT, chunk over POP if OOM.

Two forward paths share one weight-composition:
  * es_forward  : population, no_grad, per-member ternary base + shared FP residual -> (P,B,C)
  * gd_forward  : single member, base detached, grad flows ONLY to U,V (FP acts)   -> (B,C)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================ ternary ============================
def ternary_quantize(W, eps=1e-5):
    """BitNet b1.58 absmean ternary: scale = mean(|W|) over the last two dims, then
    round(W/scale) clamped to {-1,0,+1}, returned as scale * {-1,0,+1}. W: (..., O, I)."""
    scale = W.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    Wq = torch.round(W / scale).clamp_(-1.0, 1.0)
    return Wq * scale


# ============================ model ============================
class HybridViT(nn.Module):
    """A ViT whose every linear weight is `Q(W_latent) + U Vᵀ`.

    Flags:
      ternary   : ternarize the base in the forward (False -> pure FP base, an ES-FP control).
      rank      : residual rank r (0 -> no residual, i.e. a pure-ES control).
    """

    def __init__(self, image_size=28, patch_size=7, channels=1, num_classes=10,
                 dim=64, depth=2, heads=4, mlp=128, rank=4, ternary=True):
        super().__init__()
        assert image_size % patch_size == 0
        self.dim, self.depth, self.heads = dim, depth, heads
        self.patch_size, self.channels = patch_size, channels
        self.rank, self.ternary = rank, ternary
        n_patches = (image_size // patch_size) ** 2
        patch_dim = (patch_size ** 2) * channels
        self.seq = n_patches + 1                          # + cls token

        # ES-trained base parameters (latent masters; the `_w` ones are ternarized in forward)
        Pd = nn.ParameterDict()
        # GD-trained residual factors, one (U,V) pair per `_w` matrix
        Rd = nn.ParameterDict()
        self._wnames = []

        def mat(name, o, i):
            Pd[name + "_w"] = nn.Parameter(torch.empty(o, i))
            Pd[name + "_b"] = nn.Parameter(torch.zeros(o))
            self._wnames.append((name, o, i))
            if rank > 0:
                # LoRA-style init: V ~ small, U = 0  =>  residual is exactly 0 at start,
                # so the model begins as a pure ternary-ES model and GD grows the scaffold.
                Rd[name + "_U"] = nn.Parameter(torch.zeros(o, rank))
                Rd[name + "_V"] = nn.Parameter(torch.randn(i, rank) / math.sqrt(i))

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
        self.P, self.R = Pd, Rd

        for name, par in Pd.items():
            if par.dim() == 2:
                nn.init.kaiming_uniform_(par, a=math.sqrt(5))

    # -------- param groups --------
    def es_params(self):
        """The ES-trained base ParameterDict (perturbed + updated by the ES step)."""
        return self.P

    def gd_params(self):
        """The GD-trained residual factors (updated by the stateless GD step)."""
        return list(self.R.values())

    # -------- noise (ES base only) --------
    @torch.no_grad()
    def sample_noise(self, pop, rank, device, dtype):
        noise = {}
        for name, par in self.P.items():
            if name.endswith("_w"):                       # linear weight -> low-rank factors
                o, i = par.shape
                noise[name] = (torch.randn(pop, o, rank, device=device, dtype=dtype),
                               torch.randn(pop, i, rank, device=device, dtype=dtype))
            else:                                         # bias / LN / cls / pos -> dense
                noise[name] = torch.randn(pop, *par.shape, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def zero_noise(self, rank, device, dtype):
        en = {}
        for name, par in self.P.items():
            if name.endswith("_w"):
                o, i = par.shape
                en[name] = (torch.zeros(1, o, rank, device=device, dtype=dtype),
                            torch.zeros(1, i, rank, device=device, dtype=dtype))
            else:
                en[name] = torch.zeros(1, *par.shape, device=device, dtype=dtype)
        return en

    # -------- residual helper (shared by both forward paths) --------
    def _residual(self, x, name):
        """(x @ V) @ Uᵀ. Returns 0.0 if rank==0. x: (...,I) -> (...,O)."""
        if self.rank == 0:
            return 0.0
        U, V = self.R[name + "_U"], self.R[name + "_V"]
        return F.linear(x @ V, U)                          # (x@V):(...,r) ; @Uᵀ:(...,O)

    # =================== ES forward (population, no_grad) ===================
    @torch.no_grad()
    def es_forward(self, x_img, noise, sigma, rank_scale):
        P = noise["head_w"][0].shape[0]
        D, h = self.dim, self.heads

        def lin(x, name):
            W = self.P[name + "_w"]                        # (O,I) latent master
            A, B = noise[name + "_w"]                      # (P,O,r),(P,I,r)
            # per-member perturbed weight, then ternary (ternary breaks no-materialize trick)
            W_eff = W.unsqueeze(0) + (sigma * rank_scale) * torch.einsum("por,pir->poi", A, B)
            if self.ternary:
                W_eff = ternary_quantize(W_eff)            # (P,O,I)
            if x.dim() == 3:                               # (B,S,I): pop dim born here
                out = torch.einsum("bsi,poi->pbso", x, W_eff)
                out = out + self._residual(x, name).unsqueeze(0) if self.rank else out
            else:                                          # (P,B,S,I)
                out = torch.einsum("pbsi,poi->pbso", x, W_eff)
                out = out + self._residual(x, name) if self.rank else out
            out = out + self.P[name + "_b"]
            out = out + sigma * noise[name + "_b"][:, None, None, :]
            return out

        def ln(x, name):
            mu = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            xn = (x - mu) * torch.rsqrt(var + 1e-5)
            g = self.P[name + "_g"] + sigma * noise[name + "_g"]
            b = self.P[name + "_b"] + sigma * noise[name + "_b"]
            return xn * g[:, None, None, :] + b[:, None, None, :]

        x = self._patchify(x_img)                          # (B,n,patch_dim)
        x = lin(x, "patch_embed")                          # (P,B,n,D)
        B = x_img.shape[0]
        cls = (self.P["cls"] + sigma * noise["cls"])[:, None, None, :].expand(P, B, 1, D)
        x = torch.cat([cls, x], dim=2)
        pos = (self.P["pos"] + sigma * noise["pos"])
        x = x + pos[:, None, :, :]
        x = self._blocks(x, lin, ln)
        x = ln(x, "norm")
        return lin(x[:, :, 0:1, :], "head").squeeze(2)     # (P,B,C)

    # =================== GD forward (single, grad on U,V only) ===================
    def gd_forward(self, x_img, use_residual=True):
        """Single-member forward on the BASE (mean) weights. The ternary base is detached
        (frozen this phase); gradient flows only into the residual factors U,V.
        `use_residual=False` -> base-only forward (residual ignored): the adaptive controller
        uses it to measure whether the TERNARY BASE alone has improved (the harm/health signal)."""
        D, h = self.dim, self.heads
        resid = use_residual and self.rank

        def lin(x, name):
            W = self.P[name + "_w"]
            Wq = ternary_quantize(W) if self.ternary else W
            out = F.linear(x, Wq.detach())                 # base frozen
            if resid:
                out = out + self._residual(x, name)        # grad flows to U,V
            return out + self.P[name + "_b"].detach()

        def ln(x, name):
            return F.layer_norm(x, (self.dim,),
                                self.P[name + "_g"].detach(), self.P[name + "_b"].detach())

        x = self._patchify(x_img)
        x = lin(x, "patch_embed")                          # (B,n,D)
        B = x_img.shape[0]
        cls = self.P["cls"].detach()[None, None, :].expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)
        x = x + self.P["pos"].detach()[None, :, :]
        # reuse the block body with single-member (no pop dim) tensors
        for L in range(self.depth):
            p = f"block{L}"
            y = ln(x, p + "_ln1")
            Bn, Sn, _ = y.shape
            q = lin(y, p + "_q").reshape(Bn, Sn, h, D // h).transpose(1, 2)
            k = lin(y, p + "_k").reshape(Bn, Sn, h, D // h).transpose(1, 2)
            v = lin(y, p + "_v").reshape(Bn, Sn, h, D // h).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bn, Sn, D)
            x = x + lin(a, p + "_o")
            y = ln(x, p + "_ln2")
            x = x + lin(F.gelu(lin(y, p + "_fc1")), p + "_fc2")
        x = ln(x, "norm")
        return lin(x[:, 0, :], "head")                     # (B,C)

    # =================== shared pieces ===================
    def _patchify(self, x_img):
        B, ps = x_img.shape[0], self.patch_size
        xp = x_img.unfold(2, ps, ps).unfold(3, ps, ps)     # (B,C,nh,nw,ps,ps)
        xp = xp.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, self.channels * ps * ps)
        return xp                                          # (B,n_patches,patch_dim)

    def _blocks(self, x, lin, ln):
        D, h = self.dim, self.heads
        for L in range(self.depth):
            p = f"block{L}"
            y = ln(x, p + "_ln1")
            Pn, Bn, Sn, _ = y.shape
            q = lin(y, p + "_q").reshape(Pn, Bn, Sn, h, D // h).transpose(2, 3)
            k = lin(y, p + "_k").reshape(Pn, Bn, Sn, h, D // h).transpose(2, 3)
            v = lin(y, p + "_v").reshape(Pn, Bn, Sn, h, D // h).transpose(2, 3)
            a = F.scaled_dot_product_attention(q, k, v)
            a = a.transpose(2, 3).reshape(Pn, Bn, Sn, D)
            x = x + lin(a, p + "_o")
            y = ln(x, p + "_ln2")
            x = x + lin(F.gelu(lin(y, p + "_fc1")), p + "_fc2")
        return x

    # =================== fold (crystallize residual into the ternary base) ===================
    @torch.no_grad()
    def snapshot_base(self):
        """Clone the ternary latent masters so a tentative fold can be rolled back."""
        return {name: self.P[name + "_w"].data.clone() for name, _, _ in self._wnames}

    @torch.no_grad()
    def restore_base(self, snap):
        for name, t in snap.items():
            self.P[name + "_w"].data.copy_(t)

    @torch.no_grad()
    def fold_residual(self):
        """W_latent += U Vᵀ, then reset the residual to 0. The next forward re-ternarizes
        the master, so the part of the FP correction that crossed a quantization boundary is
        committed as discrete ternary flips and the sub-resolution part is discarded. Keeps
        the residual a transient SCAFFOLD instead of a permanent parallel FP model."""
        if self.rank == 0:
            return
        for name, o, i in self._wnames:
            U, V = self.R[name + "_U"], self.R[name + "_V"]
            self.P[name + "_w"].data.add_((U @ V.t()).to(self.P[name + "_w"].dtype))
            U.data.zero_()
            V.data.normal_(0, 1.0 / math.sqrt(i))


# ============================ ES loss / fitness / update ============================
def per_member_loss(logits, target):
    P, B, C = logits.shape
    flat = logits.reshape(P * B, C).float()
    tgt = target.repeat(P)                                 # pop-major: matches reshape order
    return F.cross_entropy(flat, tgt, reduction="none").reshape(P, B).mean(1)   # (P,)


def fitness_from_loss(loss):
    ranks = torch.argsort(torch.argsort(loss, descending=True))
    return ranks.float() / (loss.shape[0] - 1) - 0.5


@torch.no_grad()
def es_update(params, noise, fit, coeff, rank_scale):
    """In-place ES step on the base params (the residual is updated separately by GD).
    `_w` (ternary latent master): upd = (1/sqrt r) Σ f_p A_p B_pᵀ (Gaussian factors --
    quantization is forward-only, so the estimator is the standard one for the master)."""
    for name, par in params.items():
        if name.endswith("_w"):
            A, B = noise[name]
            upd = rank_scale * torch.einsum("p,por,pir->oi", fit, A, B)
        else:
            upd = torch.tensordot(fit, noise[name], dims=([0], [0]))
        par.data.add_(coeff * upd.to(par.dtype))
