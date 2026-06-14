"""
Stage 0.5b -- S3/S5 state-tracking trainer with LENGTH GENERALIZATION eval.

The decisive experiment (DESIGN 5, CLAUDE.md guardrail #7). Train on short
sequences, evaluate on longer ones, and watch whether accuracy holds. A true FSA
(finite-precision non-linear RNN) holds flat as length grows; a TC0 shortcut
learner collapses past the training length.

Reuses the EGGROLL ES machinery and the model verbatim; only the data source
(permutation streams from `s_n.py`) and the eval metric (per-position token
accuracy, swept over length) differ from the char-LM trainer.

Run (from the repo root):
    python tasks/state_tracking/train_sn.py --n 5 --mode tanh
    python tasks/state_tracking/train_sn.py --n 5 --mode spike --decay 1.0

Method notes baked into the defaults:
  * Run `--mode tanh` FIRST as the control (its learnable forget gate can hold
    state perfectly). If tanh generalizes and spike doesn't, the culprit is the
    threshold/dead-zone (DESIGN 6.2), not ES.
  * `decay` defaults to 1.0 here (non-leaky integrate-and-fire) -- the char-LM's
    0.9 (~7-step half-life) is FATAL for long-range tracking (DESIGN 6.4). Only
    affects spike mode (tanh uses its forget gate).
  * NoPE + a sequential recurrence => the model runs at ANY length, so eval at
    lengths far past training is well-defined.
  * COMPILE IS OFF BY DEFAULT (perf, not correctness): this task feeds many sequence
    lengths (train_lens + eval_lens + pop=1), which thrashes torch.compile's recompile
    budget for little benefit on a sequential Python recurrence. The forward is causal
    and correct in both paths (see _diag_causal.py); --compile opts in.
"""

import argparse
import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn import config                                  # noqa: E402
from spiking_m2rnn.eggroll import es_update, fitness_from_loss, per_member_loss  # noqa: E402
from spiking_m2rnn.model import SpikingM2RNN                      # noqa: E402
from s_n import SymmetricGroup, make_batch                        # noqa: E402


@torch.no_grad()
def eval_length_sweep(model, group, lengths, cfg, generators="all", batches=20):
    """Mean (over all positions) token accuracy at each eval length.

    NOTE: this average DILUTES extrapolation -- in a long eval, positions past
    train_len are pure extrapolation and dominate the count. Read it alongside the
    position profile below, which is the real length-generalization view."""
    zn = model.zero_noise(cfg.device, cfg.dtype)
    acc = {}
    for L in lengths:
        correct = total = 0
        for _ in range(batches):
            x, y = make_batch(group, cfg.batch_size, L, cfg.device, generators=generators)
            logits = model(x, zn, 0.0)[0]                         # (B,T,vocab)
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        acc[L] = correct / total
    return acc


@torch.no_grad()
def eval_position_profile(model, group, length, cfg, generators="all", batches=20):
    """Per-POSITION accuracy within one length-`length` sequence (mean over batches).

    This is the decisive length-generalization view: a model trained at train_len
    has only seen positions 0..train_len-1, so positions >= train_len here are pure
    extrapolation. A true FSA holds flat across the boundary; an overfit-to-length
    model cliffs at train_len."""
    zn = model.zero_noise(cfg.device, cfg.dtype)
    correct = torch.zeros(length, dtype=torch.float64)
    n = 0
    for _ in range(batches):
        x, y = make_batch(group, cfg.batch_size, length, cfg.device, generators=generators)
        pred = model(x, zn, 0.0)[0].argmax(-1)                   # (B,T)
        correct += (pred == y).float().sum(0).double().cpu()     # per-position hits
        n += x.shape[0]
    return correct / n                                           # (length,) accuracy per position


def _bucket_bounds(length, train_max):
    """Position buckets with the first cut at train_max, then doubling: the train
    boundary is explicit so the cliff (if any) is obvious."""
    bounds, b = [0], max(1, train_max)
    while b < length:
        bounds.append(b); b *= 2
    bounds.append(length)
    # dedupe + clamp, keep strictly increasing
    out = []
    for v in bounds:
        v = min(v, length)
        if not out or v > out[-1]:
            out.append(v)
    return out


def _fmt_sweep(acc):
    return "  ".join(f"L{L}:{a*100:5.1f}%" for L, a in acc.items())


def _fmt_profile(acc_per_pos, train_max):
    L = len(acc_per_pos)
    bounds = _bucket_bounds(L, train_max)
    segs = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = acc_per_pos[lo:hi].mean().item() * 100
        mark = "|" if lo == train_max else ""   # mark the train/extrapolation boundary
        segs.append(f"{mark}[{lo}:{hi}){m:4.0f}%")
    return " ".join(segs)


def train(group, train_lens, eval_lens, steps, eval_every, cfg, compile=False,
          generators="all", input_decay=False, mac_free=False, ternary_W=False):
    vocab = group.size
    model = SpikingM2RNN(vocab, dim=cfg.dim, depth=cfg.depth, k=cfg.k_dim, v=cfg.v_dim,
                         mlp=cfg.mlp_dim, mode=cfg.mode, threshold=cfg.threshold,
                         decay=cfg.decay, input_decay=input_decay, mac_free=mac_free,
                         ternary_W=ternary_W).to(cfg.device).to(cfg.dtype)
    model.eval(); model.requires_grad_(False)
    if compile:
        # PERF note: this task feeds many sequence lengths (train_lens + eval_lens +
        # pop=1 at eval), which thrashes torch.compile's recompile budget for little
        # benefit on a sequential Python recurrence -- hence compile is OFF by default.
        # It is correct either way (forward is causal; see _diag_causal.py); raise the
        # budget so all shapes specialise instead of falling back.
        import torch._dynamo as dynamo
        dynamo.config.recompile_limit = max(getattr(dynamo.config, "recompile_limit", 8), 128)
        dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model)
    coeff = cfg.coeff
    train_lens = sorted(set(train_lens))
    train_max = max(train_lens)
    profile_len = max(eval_lens)
    nparams = sum(p.numel() for p in model.P.values())
    ngen = group.size if generators == "all" else len(generators)
    decay_desc = ("shift-decay 2^-s + subtractive reset (MAC-free)" if mac_free
                  else "input-dependent" if input_decay else cfg.decay)
    print(f"task=S{group.n} vocab={vocab} mode={cfg.mode} params={nparams:,} "
          f"pop={cfg.pop_size} sigma={cfg.sigma} decay={decay_desc} "
          f"ternary_W={ternary_W} train_lens={train_lens} eval_lens={eval_lens} "
          f"gens={ngen} device={cfg.device}")
    chance = 1.0 / vocab
    print(f"(chance accuracy = {chance*100:.2f}%; profile '|' marks the train/extrapolation "
          f"boundary at pos {train_max}; compile={'on' if compile else 'OFF (perf)'})")

    rng = torch.Generator(device="cpu")
    for step in range(1, steps + 1):
        # variable-length training: a single fixed length overfits to that length,
        # so sample one of train_lens each step (the standard length-gen recipe).
        tl = train_lens[torch.randint(len(train_lens), (1,), generator=rng).item()]
        x, y = make_batch(group, cfg.batch_size, tl, cfg.device, generators=generators)
        noise = model.sample_noise(cfg.pop_size, cfg.rank, cfg.device, cfg.dtype)
        with torch.no_grad():
            if cfg.chunk is None:
                loss = per_member_loss(model(x, noise, cfg.sigma), y)
            else:
                parts = []
                for s in range(0, cfg.pop_size, cfg.chunk):
                    sub = {k: (tuple(t[s:s + cfg.chunk] for t in vv) if isinstance(vv, tuple) else vv[s:s + cfg.chunk])
                           for k, vv in noise.items()}
                    parts.append(per_member_loss(model(x, sub, cfg.sigma), y))
                loss = torch.cat(parts)
        fit = fitness_from_loss(loss).to(cfg.dtype)
        es_update(model.P, noise, fit, coeff, cfg.rank_scale)

        if step == 1 or step % eval_every == 0:
            acc = eval_length_sweep(model, group, eval_lens, cfg, generators)
            prof = eval_position_profile(model, group, profile_len, cfg, generators)
            print(f"step {step:05d} | loss {loss.min().item():.3f} | mean {_fmt_sweep(acc)}")
            print(f"            pos@L{profile_len}: {_fmt_profile(prof, train_max)}")


def _cli():
    ap = argparse.ArgumentParser(description="S_n state-tracking trainer (Stage 0.5b).")
    ap.add_argument("--n", type=int, default=5, help="symmetric group S_n (3 or 5)")
    ap.add_argument("--mode", choices=["spike", "tanh"], default="tanh",
                    help="run tanh control FIRST; then spike")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--train-lens", type=int, nargs="+", default=[8, 16, 24, 32],
                    help="sample one per step (variable length => length generalization). "
                         "Pass a single value to reproduce fixed-length training.")
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--generators", choices=["min", "all"], default="min",
                    help="input alphabet: 'min' = a 2-element generating set (the standard "
                         "word problem — tractable: learn 2 transitions, state still spans the "
                         "whole group); 'all' = sample from all n! perms (must learn the FULL "
                         "Cayley table — far harder, was pinning ES at chance).")
    ap.add_argument("--pop", type=int, default=config.POP_SIZE)
    ap.add_argument("--sigma", type=float, default=config.SIGMA)
    ap.add_argument("--decay", type=float, default=1.0,
                    help="spike membrane leak; 1.0 = non-leaky IF (ignored if --input-decay)")
    ap.add_argument("--input-decay", action="store_true",
                    help="spike only: learnable input-dependent decay gate (DESIGN §6.4 float "
                         "prototype; the spike analog of tanh's forget gate). Recommended for "
                         "long-range tracking — a constant leak can't both hold and forget.")
    ap.add_argument("--mac-free", action="store_true",
                    help="spike only (Stage 1a): MULTIPLY-FREE membrane — shift-decay 2^{-s_t} "
                         "(s_t in {0,1,2,3}, leak = bit-shift) + subtractive reset. Implies the "
                         "decay gate. W stays float here (ternary W is Stage 1b).")
    ap.add_argument("--ternary-w", action="store_true",
                    help="Stage 1b: ternary {-1,0,+1} state-transition W (BitNet b1.58 absmean), "
                         "per-member materialized, float latent master. Combine with --mac-free "
                         "for the full multiply-free transition. Watch σ vs bin width (DESIGN §6.2).")
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--chunk", type=int, default=config.CHUNK)
    # model size (DESIGN 8: a smaller model may give cleaner ES signal)
    ap.add_argument("--dim", type=int, default=config.DIM)
    ap.add_argument("--depth", type=int, default=config.DEPTH)
    ap.add_argument("--k", type=int, default=config.K_DIM)
    ap.add_argument("--v", type=int, default=config.V_DIM)
    ap.add_argument("--mlp", type=int, default=config.MLP_DIM)
    ap.add_argument("--compile", action="store_true",
                    help="opt into torch.compile (OFF by default for perf on this "
                         "many-length workload; correct either way — see train()).")
    args = ap.parse_args()

    cfg = dataclasses.replace(
        config.DEFAULT, mode=args.mode, pop_size=args.pop, sigma=args.sigma,
        decay=args.decay, threshold=args.threshold, batch_size=args.batch, chunk=args.chunk,
        dim=args.dim, depth=args.depth, k_dim=args.k, v_dim=args.v, mlp_dim=args.mlp,
    )
    group = SymmetricGroup(args.n)
    generators = "all" if args.generators == "all" else group.default_generators()
    train(group, args.train_lens, args.eval_lens, args.steps, args.eval_every,
          cfg, compile=args.compile, generators=generators,
          input_decay=args.input_decay, mac_free=args.mac_free, ternary_W=args.ternary_w)


if __name__ == "__main__":
    _cli()
