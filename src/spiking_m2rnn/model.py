"""
SpikingM2RNN -- a SPIKING, matrix-valued-state RNN (a spiking take on M2RNN),
trained gradient-free with EGGROLL evolution strategies.

This is the CORRECTNESS REFERENCE architecture, not the fast path:

  * FLOAT weights + EGGROLL low-rank forward. NO ternary/BitNet quantization yet,
    NO bit-packed kernel, NO on-the-fly noise. The membrane is a float accumulator
    and the leak is a real multiply. That is on purpose: this stage answers ONE
    question -- does a spiking nonlinear matrix-RNN produce a learning signal under
    ES at all? Quantization is Stage 1 (and it breaks the low-rank "never
    materialise" trick -> per-member 2-bit weights). The Triton popcount +
    in-SRAM Philox kernel is Stage 2.

  * mode="spike" is the architecture under test. mode="tanh" recovers analog M2RNN
    (eqs 10-12: Z=tanh(HW + k v^T); H=fH+(1-f)Z; y=H^T q) as a baseline, so a
    failure tells you whether it's the spiking nonlinearity or ES itself.

  * q/k/v/forget projections are computed in PARALLEL across the sequence (M2RNN's
    point); only the recurrence is sequential. GPU parallelism lives in the
    population (P) and batch (B) axes, not the sequence.

  * DEAD-ZONE WARNING: a hard spike threshold makes the loss piecewise-CONSTANT in
    every weight feeding a spike -- only the head sees a smooth landscape. That's
    why SIGMA is 0.05, not the ViT's 0.01. If the loss is flat, raise SIGMA (then
    POP, then THRESHOLD/DECAY). (DESIGN 6.2)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .eggroll import (
    eggroll_linear,
    eggroll_linear_ternary,
    eggroll_ln,
    sample_noise,
    zero_noise,
)


class SpikingM2RNN(nn.Module):
    def __init__(self, vocab, dim=config.DIM, depth=config.DEPTH, k=config.K_DIM,
                 v=config.V_DIM, mlp=config.MLP_DIM, mode=config.MODE,
                 threshold=config.THRESHOLD, decay=config.DECAY, input_decay=False,
                 mac_free=False, ternary_W=False):
        super().__init__()
        self.dim, self.depth, self.k, self.v, self.mode, self.vocab = dim, depth, k, v, mode, vocab
        self.threshold, self.decay = threshold, decay
        # input_decay (spike only): replace the constant membrane leak with a learnable,
        # input-dependent, state-INDEPENDENT decay gate -- the float prototype of DESIGN
        # 6.4's shift-decay, and the spike analog of tanh's forget gate f_t. Opt-in so the
        # Stage-0 "spike" path stays bit-identical to the frozen reference (guardrail #2).
        # mac_free (Stage 1a, spike only): make the membrane dynamics MULTIPLY-FREE --
        # the decay gate is quantized to a power of two 2^{-s_t}, s_t in {0,1,2,3} (the
        # leak is a bit-shift, not a float multiply), and the reset is subtractive
        # (U -= theta*S) rather than a masked zero. Implies the decay gate. W stays float
        # here (so the accumulator is still float via `trans`); ternary W + integer
        # membrane is Stage 1b.
        self.mac_free = mac_free and mode == "spike"
        self.input_decay = (input_decay or self.mac_free) and mode == "spike"
        # ternary_W (Stage 1b): quantize the state-transition W to {-1,0,+1} in the
        # forward (per-member, materialized), keeping a float latent master that ES
        # updates as usual. This is the BitNet core: trans = W_ternary . S(binary).
        # Opt-in; Stage-0 path unchanged. (DESIGN 6.6 -- breaks the no-materialize trick.)
        self.ternary_W = ternary_W

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
            if self.input_decay:
                mat(p + "_d", 1, dim)                  # input-dependent decay-gate logits (spike)
            mat(p + "_W", v, v)                        # state transition (V x V) -- the BitNet-able matrix
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

    # ---- per-step noise (delegates to the shared EGGROLL machinery) ----
    @torch.no_grad()
    def sample_noise(self, pop, rank, device, dtype):
        return sample_noise(self.P, pop, rank, device, dtype)

    @torch.no_grad()
    def zero_noise(self, device, dtype, rank=config.RANK):
        return zero_noise(self.P, rank, device, dtype)

    def _lin(self, x, name, noise, sigma):
        A, B = noise[name + "_w"]
        return eggroll_linear(x, self.P[name + "_w"], A, B,
                              bias=self.P[name + "_b"], bias_noise=noise[name + "_b"], sigma=sigma)

    def _ln(self, x, name, noise, sigma):
        return eggroll_ln(x, self.P[name + "_g"], noise[name + "_g"],
                          self.P[name + "_b"], noise[name + "_b"], sigma=sigma)

    def forward(self, idx, noise, sigma=config.SIGMA, return_fire=False):
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
            d_t = None
            if self.mode == "spike":
                Q  = (Q  > 0).to(x.dtype)                     # spike-encode q/k/v (threshold 0; bias = -threshold)
                Kp = (Kp > 0).to(x.dtype)
                Vp = (Vp > 0).to(x.dtype)
                f_t = None
                if self.input_decay:                          # learnable input-dependent leak
                    raw = self._lin(xn, p + "_d", noise, sigma)                  # (P,B,T,1)
                    if self.mac_free:
                        # quantize to a shift amount s_t in {0,1,2,3} -> decay = 2^{-s_t}
                        # (s_t=0 hold, s_t=3 fast forget); the leak is a bit-shift in-kernel.
                        s_t = torch.round(torch.sigmoid(raw) * 3.0).clamp_(0.0, 3.0)
                        d_t = torch.pow(2.0, -s_t)
                    else:
                        d_t = torch.sigmoid(raw)                                 # continuous in (0,1)
            else:                                             # analog M2RNN
                f_t = torch.sigmoid(self._lin(xn, p + "_f", noise, sigma))   # (P,B,T,1)

            state = torch.zeros(Pn, Bn, k_, v_, device=x.device, dtype=x.dtype)   # H (analog) | S_prev (spiking)
            mem   = torch.zeros(Pn, Bn, k_, v_, device=x.device, dtype=x.dtype)   # U (spiking only)
            ys = []
            for t in range(Tn):
                kt = Kp[:, :, t, :]; vt = Vp[:, :, t, :]; qt = Q[:, :, t, :]
                outer = torch.einsum("pbk,pbv->pbkv", kt, vt)                      # k_t (x) v_t
                if self.ternary_W:                                                 # BitNet transition (Stage 1b)
                    A, Bw = noise[p + "_W_w"]
                    trans = eggroll_linear_ternary(state, self.P[p + "_W_w"], A, Bw,
                                                   bias=self.P[p + "_W_b"],
                                                   bias_noise=noise[p + "_W_b"], sigma=sigma)
                else:
                    trans = self._lin(state, p + "_W", noise, sigma)               # transition on PREVIOUS state
                if self.mode == "spike":
                    decay = d_t[:, :, t, :].unsqueeze(-1) if self.input_decay else self.decay  # (P,B,1,1) | scalar
                    mem = decay * mem + trans + outer
                    S   = (mem > self.threshold).to(x.dtype)
                    if self.mac_free:
                        mem = mem - self.threshold * S                            # subtractive reset
                    else:
                        mem = mem * (1.0 - S)                                      # hard reset
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
