# SPIKE-M2RNN — Design & Context

> Deep reference document. Read this to understand *why* the project is shaped the
> way it is. For day-to-day operating rules and conventions, see `CLAUDE.md` (which
> is intentionally short). This document is the source of truth for design intent;
> `CLAUDE.md` is the source of truth for "how we work in this repo."

---

## 1. What this is, in one paragraph

A **spiking, ternary (BitNet-style), matrix-valued-state RNN** — a spiking
reinterpretation of **M2RNN** — trained **gradient-free with EGGROLL evolution
strategies**. The bet: three individually *gradient-hostile* ingredients (a hard
spike threshold, ternary weights, a non-linear recurrence) become tractable
*together* precisely because ES needs no gradients. There is no surrogate gradient,
no straight-through estimator, and no backprop-through-time. The architectural
payoff we are chasing is **state-tracking expressivity** (M2RNN's reason for being)
delivered on a **multiply-free** substrate.

The combination is the thesis. None of the three pieces is novel alone; the claim is
that ES is the key that makes all three practical at once, and that a thresholded
(spiking) non-linear matrix-RNN is a clean finite-state automaton that should
length-generalize on hard state-tracking.

---

## 2. Background

### 2.1 EGGROLL — Evolution Strategies at hyperscale (arXiv:2511.16652)
What we use from it:
- **Low-rank perturbations.** Each population member perturbs a weight `W` by a
  rank-`r` factor pair `(A, B)`: effective weight `W + (σ/√r)·A Bᵀ`. The forward is
  computed as one shared base GEMM `F.linear(x, W)` plus a cheap low-rank
  correction, so a perturbed weight is **never materialized** per member.
- **Rank-based fitness shaping** + the update `coeff · Σ_p f_p A_p B_pᵀ` with
  `coeff = LR / (POP·σ)`.
- **Integer-seed noise.** EGGROLL samples per-member noise from simple incremental
  integer seeds (per the user's reading of the paper) — this is what makes the
  Stage-2 in-kernel noise regeneration straightforward (see §7).

Our working EGGROLL reference is the **MNIST ViT** (`reference/` — converged to
~80% on MNIST). The Stage-0 model reuses its machinery verbatim:
`eggroll_linear`, `eggroll_ln`, `sample_noise`, `fitness_from_loss`, and the
update loop.

### 2.2 M2RNN — Non-Linear RNN with matrix-valued states (arXiv:2603.14360)
The architecture we are adapting. Core recurrence (their eqs 10–12, single head):

```
Z_t = tanh( H_{t-1} W + k_t v_tᵀ )          # non-linear state transition
H_t = f_t H_{t-1} + (1 - f_t) Z_t           # gated convex update
y_t = H_tᵀ q_t  +  w_r ⊙ v_t                # query readout + residual
```

with `H ∈ R^{K×V}` (matrix state), `W ∈ R^{V×V}` **input-independent**, and
`q_t,k_t ∈ R^K`, `v_t ∈ R^V`, `f_t ∈ [0,1]` produced by input-only projections.
Their models use `K=64, V=16` (multi-value formulation).

Findings that matter to us:
- **State size, not non-linearity, was the historical bottleneck** for non-linear
  RNNs. The matrix state fixes capacity; the non-linearity buys state-tracking.
- **The forget gate `f_t` is input-dependent but state-INDEPENDENT.** This lets all
  pre-recurrence projections be computed in parallel across the sequence and
  controls gradient flow — it is *not* what makes the recurrence parallel.
- **M2RNN is sequential over time.** The `tanh` breaks associative-scan
  parallelization; their forward kernel (Algorithm 1) has a literal `for t` loop.
  Its hardware efficiency comes from a **batch-independent GEMM shape** (`H_{t-1}W`
  is `(K,V,V)`, independent of batch) that fills tensor cores without padding — and
  from using the layer **sparingly** in hybrids.
- **Expressivity anchor:** state-tracking lives in NC1 (e.g. the S5 word problem is
  NC1-complete); Transformers and diagonal/positive-eigenvalue linear RNNs are stuck
  in TC0. Merrill (2019), cited therein: a **finite-precision non-linear RNN is
  exactly a finite-state automaton** and recognizes all regular languages. This is
  the theoretical license for the spiking design — a thresholded recurrence is
  finite-precision and therefore in-principle expressive for regular-language state
  tracking. Crisp discrete states may also length-generalize *better* than analog
  ones (no float drift over long sequences — the failure mode M2RNN shows for Gated
  DeltaProduct).

### 2.3 The spiking reinterpretation (this project)
We replace M2RNN's `tanh` with a **hard spike threshold + reset**, and (eventually)
make `W` ternary. The Stage-0 recurrence (variant A — see §6.1):

```
outer = k_spk ⊗ v_spk                        # {0,1} outer product (AND)
trans = linear(S_prev, W)                    # transition on PREVIOUS spikes; W ternary in Stage 1
U     = decay·U + trans + outer              # leaky integrate (membrane)
S     = 1[ U > θ ]                           # fire (the non-linearity)
U     = U · (1 - S)                          # hard reset where fired
y     = Σ_k S[k,:] · q_spk[k]                # query-gated count readout
```

Mapping to M2RNN: `tanh` → Heaviside; analog state `H` → (membrane `U`, spikes `S`);
the transition mixes the **1-bit** state `S` rather than the analog state. The
forget gate becomes the leak `decay` (a constant at Stage 0; see §6.4 for the
input-dependent, multiply-free upgrade).

---

## 3. Why ES is the keystone (the unifying argument)

Each ingredient is a known backprop pathology, and ES removes all three at once:
- **Spike threshold** → normally trained with surrogate gradients. ES: it's just a
  forward op.
- **Ternary weights** → normally trained with a straight-through estimator. ES:
  quantization is just a forward op; ES optimizes the Gaussian-smoothed *true*
  discrete forward, which is arguably more honest than STE.
- **Non-linear recurrence** → normally BPTT with vanishing/exploding gradients and
  O(T) activation memory. ES: forward-only, so **memory is O(1) in sequence
  length** (carry the state, accumulate loss, free intermediates) and there are no
  gradient-stability issues.

Sequential-in-time is therefore acceptable: there is no backward pass, the
population (P) and batch (B) axes provide the GPU parallelism, and M2RNN itself is
sequential-in-time anyway. The honest caveat is that **ES gradient-estimate variance
grows with parameter count** — that is the bet EGGROLL makes (low-rank + huge P),
and this architecture is a stress test of it, not a hedge.

---

## 4. Current status (empirical)

**Stage 0 is implemented and CONVERGING on char-level Shakespeare.** This validates
the *pipeline* (ES + spiking + matrix recurrence learns end-to-end). It does **not**
validate state-tracking — that verdict comes from §6.2.

Config: `mode=spike`, params **224,449**, `pop=512`, `sigma=0.05`, `block=64`,
`dim=128`, `K=V=64`, `device=cuda`, `dtype=float16`.

| step | val (nats) | val (bpc) | firing |
|-----:|-----------:|----------:|-------:|
| 1    | 4.087      | 5.90      | 0.188  |
| 100  | 3.324      | 4.79      | 0.247  |
| 500  | 3.132      | 4.52      | 0.314  |
| 1000 | 2.997      | 4.32      | 0.331  |
| 1500 | 2.934      | 4.23      | 0.330  |
| 1800 | 2.960      | 4.27      | 0.339  |

Reading it: uniform-char baseline is `log2(65) ≈ 6.02` bpc; Stage 0 starts at chance
and reaches ~4.23 bpc while still descending, so it is learning real character
statistics. It will **not** approach Adam-trained nanoGPT (~1.4–1.5 bpc) — that is
expected for ES at this scale and is not a defect. **fp16 trains stably here** (the
old fp16-softmax worry is moot: there is no attention softmax, only the final CE).
Firing rate self-stabilized in a healthy band (~33%).

---

## 5. Staging roadmap

Climb this ladder. Do **not** jump to the kernel before Stage 1 numerically
validates against Stage 0. Each rung isolates one source of failure.

| Stage | What | Status | Isolates |
|------:|------|--------|----------|
| **0** | Float weights, EGGROLL low-rank forward, float membrane + multiplicative decay, spiking dynamics, char Shakespeare | **DONE / converging** | "does a spiking matrix-RNN learn under ES?" |
| **0.5** | (a) analog `tanh` M2RNN baseline as a control; (b) **S3/S5 state-tracking with length generalization** | TODO | the architecture's actual reason for being |
| **1a** | MAC-free conversion: integer membrane, shift-based leak `U - (U>>n)`, soft/subtractive reset, input-dependent shift-decay | TODO | the multiply-free claim |
| **1b** | Ternary quantization of `W` (and other matrices): float latent master, quantize-in-forward, **per-member 2-bit materialization** (breaks the no-materialize trick — see §6.6), σ tuned to bin width | TODO | the BitNet cost |
| **2** | Triton kernel: bit-packed ternary × binary spikes via AND+popcount, **in-SRAM Philox noise**, fused recurrence step, bit-exact vs Stage 1 | TODO | throughput / scale |

State-tracking (Stage 0.5b) is the scientifically decisive experiment. Shakespeare
only earns the right to run it.

---

## 6. Design decisions & rationale (the "why" that's easy to lose)

### 6.1 The transition-domain fork
The transition `W` can act on either:
- **(A) the spikes `S`** (1-bit) → maps to a popcount kernel, is a clean FSA, and is
  what Stage 0 uses. Theory (Merrill 2019) says a finite-precision threshold
  recurrence is FSA-expressive, so this *should* suffice for state-tracking.
- **(B) the membrane `U`** (integer) → richer per-step mixing, but it's a ternary ×
  int8 add-only GEMM (BitNet b1.58 style), **not** popcount.

Try A first; fall back to B only if state-tracking accuracy stalls because the
1-bit-before-mix step is too lossy.

### 6.2 The dead-zone / `sigma` issue — THE recurring gotcha
A hard threshold makes the loss **piecewise-constant** in every weight that feeds a
spike; only the head sees a smooth landscape. A perturbation that flips no spikes
returns zero fitness signal. This is why `SIGMA=0.05` (not the ViT's `0.01`). In
Stage 1 this becomes the **ternary bin-width** problem: σ must be comparable to the
distance to the nearest quantization boundary, or most weights get no signal. If
spike-mode is flat while the `tanh` baseline descends, raise `SIGMA` first, then
`POP`. This is the single most likely thing to bite.

### 6.3 Firing-rate homeostasis
Keep firing roughly 10–35% (we observed ~33%). Knobs: `THRESHOLD`, `DECAY`, `SIGMA`.
A network that goes silent or saturates carries no signal. Consider adding a
firing-rate term to the fitness later; it's cheap to bolt onto rank-based shaping.
NOTE: the eval-time firing readout currently forces a `torch.compile` graph break
via `.item()` — gate it behind a flag in the hot path (see §9).

### 6.4 The `decay=0.9` caveat
Constant `0.9` gives a ~7-step half-life — fine for char-LM (local statistics), but
**fatal for long-range state-tracking** (it erases state set early unless a fragile
self-excitation loop refreshes it). For Stage 0.5b, replace it with the multiply-free
analog of M2RNN's per-head forget gate: an **input-dependent shift amount**
`s_t ∈ {0,1,2,3}`, `decay = 2^{-s_t}`, applied as `U - (U >> s_t)`. `s_t=0` is
perfect hold; large `s_t` is fast forget. State-independent, input-dependent,
MAC-free.

### 6.5 Other choices
- **NoPE** (no positional embeddings): the recurrence provides order, matching M2RNN.
- **Antithetic sampling** (variance reduction): the mirror of `σ A Bᵀ` is just `-A`,
  so use `(A,B)` and `(-A,B)` as a pair. Bake the even/odd convention into the seed
  map *before* writing the kernel (§7).
- **Spiking core, analog residual:** embeddings, residual stream, and head stay
  float (the head must emit real logits for CE). Spikes live only inside the
  recurrence. The M2RNN output gate `g_t` and residual `w_r⊙v_t` are currently
  omitted (the residual's job is gradient flow, moot under ES).

### 6.6 The EGGROLL × ternary tension (read before Stage 1)
EGGROLL's "never materialize" trick requires the forward to be **linear in the
perturbation**. `quantize(W + σ A Bᵀ)` is not, so per-member quantization forces a
per-member weight — the O(P·m·n) blow-up EGGROLL avoids. **For ternary this is
cheap**: 2 bits/weight (a 64×64 layer × P=4096 ≈ 4 MB), so materialize the per-member
ternary transiently (construct from the regenerated rank-1 perturbation, quantize,
use, discard) while keeping **one float latent master weight** (QAT-style) that ES
updates.

Consequence — **the perf model changes**: there is no longer a shared base GEMM +
cheap correction; you do P full (bit-)matmuls. Speed comes from popcount cheapness +
no-HBM-noise + huge-P parallelism, *not* from EGGROLL's GEMM-sharing. And **low-rank's
role shrinks**: with quantization + regen it is no longer a forward/memory trick; it
survives only as (a) the perturbation distribution EGGROLL argues is sufficient and
(b) a ~32× smaller RNG payload (`o·r + i·r` vs `o·i` numbers at r=1). Keep rank-1 for
the search-distribution reason; raising rank is NOT free (scales regen and update).

---

## 7. Stage-2 kernel invariants (for the Triton work)

The Stage-2 idea: push EGGROLL's seed-based noise reconstruction (classic in
distributed ES) **down into the kernel** — regenerate `A_p, B_p` in SRAM via a
counter-based RNG keyed by the integer seed, so the population `P` decouples from
HBM capacity *and* bandwidth and becomes a pure time knob. The noise has near-zero
reuse, so regenerating (compute-cheap) beats streaming (bandwidth-bound).

Composition with M2RNN's Algorithm 1 + per-member quantization:
```
for n in heads (over SMs):
  load base W[n] -> SRAM                                   # once, reused
  for p in population (over CTAs):
    A_p,B_p = philox(key=(step,p), offset=param_id)        # in SRAM, never HBM
    W_eff   = quantize(W[n] + sigma * A_p @ B_pᵀ)          # SRAM
    Wpos,Wneg = pack_bitplanes(W_eff)                      # SRAM, reused across t,b
    for b in batch:
      U = 0 (K×V) in SRAM
      for t in 0..T-1:   # sequential; popcount recurrence step
        out[o] = Σ_w popcount(Wpos[o,w] & s[w]) - popcount(Wneg[o,w] & s[w])
        ...
```

**Hard invariants** (violating any of these silently biases the ES estimator):
1. **Forward and update kernels must regenerate bit-identical noise.** Both in
   Triton/Philox, member key `(step, p)`, distinct per-parameter counter offsets →
   they agree by construction.
2. **Use Gaussian factors.** EGGROLL samples `randn`; the update's `1/σ` scaling
   assumes `N(0,I)`. Add a uniform→normal transform (Box–Muller / inverse-CDF) in
   the kernel; raw Philox is uniform.
3. **Bake antithetic pairing into the seed map up front** (even/odd member = ±A).
4. **Philox, not LCG**; distinct counter offsets so `A`, `B`, and different matrices
   decorrelate.
5. **Triton-Philox ≠ `torch.randn`.** Cross-framework determinism is NOT free, so
   the validation path must be fully materialized in torch (small P); production
   trusts kernel↔kernel self-consistency only.
6. **Keep the Stage-0/Stage-1 materialized reference path** for numerical
   equivalence checks at every stage. The core EGGROLL math was verified equivalent
   to brute-force weight materialization to ~1e-15 in float64 — preserve a test that
   asserts this.

---

## 8. Open questions / risks

- **Control:** does spike-mode track the analog `tanh` M2RNN baseline? If tanh learns
  and spike doesn't, the culprit is the threshold (→ §6.2), not ES.
- **The real question:** does spike-mode length-generalize on S3/S5 past the training
  length (reproduce M2RNN's perfect-generalization plot)?
- **Variant A vs B:** does the 1-bit-state transition preserve state-tracking, or is
  membrane-transition (B) required?
- **ES scaling:** how far do `POP`/model size go before variance dominates? 224k
  params at `POP=512` is already ~2× the ViT with far fewer samples — expect to need
  larger `POP` (chunked) and/or a smaller model for clean signal.
- **Precision:** fp16 holds at Stage 0; re-check at Stage 1 (integer membrane) and
  Stage 2.
- **σ vs bin width** in Stage 1 — the dead-zone, now from quantization.

---

## 9. Known issues / perf notes (carry-over from the Stage-0 run)
- `torch.compile` recompiles repeatedly: (a) the eval firing readout `S.mean().item()`
  forces a graph break — gate it out of the compiled path or compute firing without
  `.item()` and log outside; (b) `_lin`/`_ln` recompile across layers due to
  shape (`mlp=256` vs `dim=128`) and the layer-name string varying — consider
  marking dynamic or specializing. These are perf-only; correctness is unaffected
  (Stage 0 converged with them present).
- `Not enough SMs to use max_autotune_gemm` is benign on the current device.

---

## 10. References
- **EGGROLL:** *Evolution Strategies at the Hyperscale*, arXiv:2511.16652.
- **M2RNN:** *Non-Linear RNNs with Matrix-Valued States for Scalable Language
  Modeling*, arXiv:2603.14360. Code: github.com/open-lm-engine (lm-engine,
  accelerated-model-architectures).
- **nanoGPT** (Karpathy) — char-Shakespeare task convention (vocab 65, 90/10 split).
- Merrill (2019), *Sequential Neural Networks as Automata* (finite-precision
  non-linear RNN ≡ FSA); Merrill et al. (2024), *The Illusion of State in SSMs*
  (TC0 vs NC1 state-tracking).
