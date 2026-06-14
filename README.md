# SPIKE-M2RNN — experiment launch guide

A **spiking, ternary (BitNet), matrix-valued-state RNN** — a spiking take on
**M2RNN** — trained **gradient-free with EGGROLL evolution strategies**. The goal is
**state-tracking expressivity on a multiply-free substrate**.

- **Why it's shaped this way:** [docs/DESIGN.md](docs/DESIGN.md) (source of truth for intent).
- **How we work / guardrails:** [CLAUDE.md](CLAUDE.md).
- This file = **how to launch each stage's experiment**. Climb the ladder in order;
  do **not** skip a rung (each must validate against the one below it).

## Setup
```bash
pip install torch                          # + torchvision only for the MNIST reference
# tiny-shakespeare data: put input.txt at the path you pass with --data
```
All training is gradient-free (no `.backward()`): the model runs under `no_grad`,
params are frozen, the ES update is an explicit in-place step. GPU is recommended;
the modular code also runs on CPU (small configs) for tests.

> Run trainers **from the `src/` directory** so `python -m spiking_m2rnn.<...>` resolves.

---

## Stage 0 — pipeline smoke test (char Shakespeare) — ✅ DONE / converging
**Tests:** does a spiking matrix-RNN learn end-to-end under ES at all? (Not state-tracking.)

```bash
cd src
python -m spiking_m2rnn.train --data ../path/to/input.txt
```
Common knobs: `--steps 3000 --eval-every 100 --pop 512 --sigma 0.05 --block 64`,
`--chunk 128` (population slicing for OOM), `--no-compile` (skip `torch.compile`).

**Expect** (see [logs/Stage_0.log](logs/Stage_0.log)): val 5.90 → ~4.23 bpc, firing
self-stabilizing **~33%**. It will **not** reach Adam nanoGPT (~1.4 bpc) — expected for
ES at this scale. Watch the `fire` readout: silent/saturated = no signal (keep ~10–35%).

---

## Stage 0.5 — the real validation
### (a) `tanh` baseline control — ✅ runnable now
**Tests:** isolates the spiking nonlinearity from ES. Recovers analog M2RNN
(`Z=tanh(HW+kvᵀ)`). If tanh learns and spike doesn't, the culprit is the
threshold/dead-zone (DESIGN §6.2), **not** ES.
```bash
cd src
python -m spiking_m2rnn.train --mode tanh --data ../path/to/input.txt
```

### (b) S3/S5 state-tracking with length generalization — ✅ runnable
**Tests:** the architecture's actual reason for being. Predict the running product of
a permutation stream; **train at short T, eval at longer T**. S5 is non-solvable ⇒
NC1-complete (a Transformer/diagonal-SSM cannot track it; a finite-precision
non-linear RNN can).

```bash
python tasks/state_tracking/s_n.py                          # generator self-check
python tasks/state_tracking/train_sn.py --n 3 --mode tanh   # S3 CONTROL first (confirm it learns)
python tasks/state_tracking/train_sn.py --n 5 --mode tanh   # then climb to S5
python tasks/state_tracking/train_sn.py --n 5 --mode spike --decay 1.0
```
Input alphabet defaults to a **2-element generating set** (`--generators min`, the
tractable standard word problem); `all` forces learning the full Cayley table and pins
ES at chance. ES is sample-hungry and **S5 is the hardest group**, so confirm learning
on S3 before climbing. Trains on **variable lengths** (`--train-lens 8 16 24 32`) and
prints two eval views: `mean Lk` (averaged — dilutes extrapolation) and the decisive
`pos@Lk` per-position profile, where `|` marks the train/extrapolation boundary — flat
across `|` = generalizing, a cliff = overfit to length. Full knob list +
difficulty-ladder tips: [tasks/state_tracking/README.md](tasks/state_tracking/README.md).

---

## Stage 1a — MAC-free recurrence dynamics — 🛠️ IN PROGRESS
Multiply-free membrane: input-dependent **shift-decay** `2^{-s_t}` (`s_t∈{0,1,2,3}`,
leak = bit-shift) + **subtractive reset**. `W` stays float here (so `trans` keeps the
accumulator float); integer membrane lands once `W` is ternary (1b). Opt-in, validated
to learn S3 length-gen; Stage-0 `spike` path stays bit-identical (equivalence test).
```bash
# float prototype of the decay gate (the spike analog of tanh's forget gate):
python tasks/state_tracking/train_sn.py --n 3 --mode spike --input-decay <model/ES flags>
# MAC-free dynamics (shift-decay + subtractive reset):
python tasks/state_tracking/train_sn.py --n 3 --mode spike --mac-free   <model/ES flags>
```
`--input-decay` (continuous gate) already gives **perfect, stable S3 length-gen** —
on par with tanh. `--mac-free` is the multiply-free version of the same.

## Stage 1b — ternary `W` (BitNet) — 🛠️ IN PROGRESS
Float latent master + quantize-in-forward (BitNet b1.58 absmean → `{−1,0,+1}`) +
**per-member materialization** (ternary breaks EGGROLL's no-materialize trick, so each
member's perturbed `W` is built, quantized, used — DESIGN §6.6). The ES update on the
float master is unchanged. The materialized path with quant *off* is bit-identical to
`eggroll_linear` (equivalence test, guardrail #2).
```bash
# full multiply-free transition: ternary W . binary spikes, shift-decay, subtractive reset
python tasks/state_tracking/train_sn.py --n 3 --mode spike --mac-free --ternary-w <flags>
```
Currently ternarizes the transition `_W` (the BitNet core). Watch **σ vs bin width**
(DESIGN §6.2). Next: extend to other matrices, then Stage 2 kernel.

## Stage 2 — Triton kernel — 🚧 TODO
Bit-packed ternary × binary spikes via AND+popcount, **in-SRAM Philox noise**, fused
recurrence. **Must be bit-exact vs Stage 1** (kernel invariants: DESIGN §7). (No
launch command yet.)

---

## Reference experiment (not a stage) — EGGROLL ViT on MNIST
The canonical ES machinery this project mirrors; confirms EGGROLL converges.
```bash
python reference/eggroll_vit_mnist.py     # ~81% in 5 epochs; see logs/eggroll_vit_mnist.log
```

## Tests — cross-stage numerical equivalence
The refactor guard: the modular package must be **bit-identical** to the frozen
single-file `Stage_0.py` reference (preserve this path at every stage — guardrail #2).
```bash
python -m pytest tests/ -q               # or: python tests/test_equivalence.py
```
