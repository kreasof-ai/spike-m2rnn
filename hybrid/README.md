# hybrid/ — ES + GD hybrid (ternary base + low-rank FP residual)

A **fork** of the spiking-M2RNN project. Self-contained: imports nothing from
`src/spiking_m2rnn`, touches no shared state. Substrate is the EGGROLL ViT
(`reference/eggroll_vit_mnist.py`), MNIST/CIFAR — chosen to isolate the ES+GD
mechanism from the spiking dead-zone.

## Idea

Partition parameter space **by differentiability** and give each optimizer the part it's
good at:

```
W_eff = Q(W_latent)   # ternary {-1,0,+1}, trained GRADIENT-FREE by ES
      + U @ Vᵀ        # rank-r FP residual, trained by GRADIENT DESCENT
```

- **ES owns the non-differentiable part** — the quantization decisions `Q(·)`. ES doesn't
  care that `round/clamp` has zero gradient a.e.; that's its whole point. No STE.
- **GD owns the smooth part** — a small FP residual that *is* differentiable, so we get
  exact gradients in that subspace instead of paying ES's `O(1/√POP)` variance for it.

This is **QLoRA inverted**: QLoRA freezes a quantized base and trains an FP LoRA; here the
base is *also* trained — by the one method that trains a quantized base without a surrogate.

Training is **block coordinate descent**: alternate (ES updates base, residual frozen) and
(GD updates residual, base frozen) on the same batch.

## Two deliberate consequences

1. **FP activations** throughout — GD can't cross a hard nonlinearity without a surrogate,
   so multi-layer GD forces FP acts. This fork **trades multiply-free activation for GD
   sample-efficiency**. The win is "weights mostly 1.58-bit and gradient-trainable," *not*
   "multiply-free." Different value proposition than the main thesis — by design.
2. **Stateless GD** — ES moves the base *underneath* the residual between steps, so Adam's
   moment buffers go stale. The GD step is stateless: `rmsnorm` (RMS-normalized SGD, default),
   `sign` (signSGD), or `muon` (Newton-Schulz, momentum-free).

## The risk this is built to watch: division of labor

GD gets exact gradients every step; ES gets noisy ones. GD wins the race and will absorb
whatever it can express — so a too-large residual silently *becomes* the model and the
ternary base goes vestigial (opposite of "most params ternary, load-bearing").

Two defenses, and a direct measurement:

- **Small `--rank`** (it's scaffolding: 2–8). Init is LoRA-style (`U=0`) so the residual
  starts at exactly zero and the model begins as pure ternary-ES.
- **`--fold-every N`** — *crystallize*: `W_latent += U Vᵀ`, reset residual to 0. The next
  forward re-ternarizes, committing the correction as discrete ternary flips wherever it
  crossed a quantization boundary; the sub-resolution part is discarded. Keeps the residual
  a transient scaffold, not a permanent parallel FP model.
- **Measurement**: every eval prints `ternary-only` accuracy = the model with the residual
  **zeroed**. Small gap to `val` ⇒ base is load-bearing (good). Large gap ⇒ GD ate the model
  → lower `rank` or turn on fold.

## Brackets (controls the hybrid must beat)

| command | what it is |
|---|---|
| `--rank 0` | pure-ES ternary (no residual, no GD) |
| `--no-ternary` | pure-ES FP (the original EGGROLL ViT regime) |
| `--rank 0 --no-ternary` | plainest ES baseline |
| `--pure-gd` | standard full backprop FP (GD upper bound, ES off) |
| *(defaults)* | the hybrid: ternary base + rank-4 FP residual, interleaved |

## Run

```bash
python hybrid/train.py --dataset mnist                       # the hybrid
python hybrid/train.py --dataset mnist --rank 0              # pure-ES ternary control
python hybrid/train.py --dataset mnist --pure-gd            # GD upper bound
python hybrid/train.py --dataset mnist --fold-every 50       # crystallize scaffold
python hybrid/train.py --dataset cifar10 --steps 4000
```

Defaults are CPU-safe-ish (`--pop 1024 --chunk 512 --dtype float32`); on GPU raise `--pop`
and consider `--dtype bfloat16`. Key knobs: `--rank`, `--gd-lr`, `--gd-opt`, `--fold-every`,
plus the usual ES `--sigma/--lr/--pop`.

## The headline result we're hunting

The hybrid reaches a target accuracy in **far fewer ES evals** (`es-evals`, the cost metric —
ES forwards dominate) than pure-ES ternary, *while* `ternary-only` accuracy stays close to
`val` (the base stays load-bearing). If only `val` moves and `ternary-only` lags, the residual
is carrying the model — tighten `rank` / enable fold.

## Files

- `hybrid_vit.py` — `HybridViT` (`es_forward` population path, `gd_forward` single/grad path,
  `fold_residual`), ternary quantize, ES loss/fitness/update. Self-contained.
- `train.py` — interleaved trainer, stateless GD step, ablation eval, baselines, CLI.
