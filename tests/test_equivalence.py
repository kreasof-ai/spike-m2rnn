"""
Refactor guard: the modular `src/spiking_m2rnn` package must be NUMERICALLY
IDENTICAL to the frozen single-file `Stage_0.py` reference.

This protects the behaviour-preserving split (CLAUDE.md) and, per guardrail #2,
keeps a materialized reference path that later stages (ternary, kernel) validate
against. Runs on CPU in float64 for a tight tolerance.

    python -m pytest tests/test_equivalence.py        # or: python tests/test_equivalence.py
"""

import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


def _load_stage0():
    """Import the frozen reference as a standalone module (it has no package deps)."""
    path = os.path.join(SRC, "spiking_m2rnn", "Stage_0.py")
    spec = importlib.util.spec_from_file_location("stage0_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# small, CPU-friendly, float64 config shared by both paths
VOCAB, POP, RANK, B, T = 11, 6, 1, 2, 5
DIM, DEPTH, K, V, MLP = 16, 2, 8, 8, 24
SIGMA, THRESH, DECAY = 0.05, 1.0, 0.9
DTYPE, DEVICE = torch.float64, torch.device("cpu")


def _copy_params(dst, src):
    """Copy base weights src->dst so both models share identical parameters."""
    with torch.no_grad():
        for name in dst.P.keys():
            dst.P[name].copy_(src.P[name])


def _build_models(mode):
    from spiking_m2rnn.model import SpikingM2RNN as ModularModel

    ref_mod = _load_stage0()
    # force the reference's module-level config to match our tiny test config
    ref_mod.SIGMA = SIGMA
    ref_mod.THRESHOLD = THRESH
    ref_mod.DECAY = DECAY

    torch.manual_seed(0)
    ref = ref_mod.SpikingM2RNN(VOCAB, dim=DIM, depth=DEPTH, k=K, v=V, mlp=MLP,
                               mode=mode).to(DEVICE).to(DTYPE)
    new = ModularModel(VOCAB, dim=DIM, depth=DEPTH, k=K, v=V, mlp=MLP, mode=mode,
                       threshold=THRESH, decay=DECAY).to(DEVICE).to(DTYPE)
    new.eval(); ref.eval()
    new.requires_grad_(False); ref.requires_grad_(False)
    _copy_params(new, ref)
    return ref, new, ref_mod


def _sample_shared_noise(params):
    torch.manual_seed(1234)
    noise = {}
    for name, par in params.items():
        if name.endswith("_w"):
            o, i = par.shape
            noise[name] = (torch.randn(POP, o, RANK, dtype=DTYPE),
                           torch.randn(POP, i, RANK, dtype=DTYPE))
        else:
            noise[name] = torch.randn(POP, *par.shape, dtype=DTYPE)
    return noise


def _check_forward(mode):
    ref, new, _ = _build_models(mode)
    idx = torch.randint(0, VOCAB, (B, T))
    noise = _sample_shared_noise(ref.P)

    lo_ref = ref(idx, noise, SIGMA)
    lo_new = new(idx, noise, SIGMA)
    err = (lo_ref - lo_new).abs().max().item()
    assert err < 1e-10, f"[{mode}] forward mismatch: max abs err {err}"
    return err


def test_forward_spike():
    err = _check_forward("spike")
    print(f"spike forward max abs err: {err:.2e}")


def test_forward_tanh():
    err = _check_forward("tanh")
    print(f"tanh forward max abs err: {err:.2e}")


def test_es_update_matches():
    """One full ES step (loss -> fitness -> update) must move params identically."""
    from spiking_m2rnn.eggroll import es_update, fitness_from_loss, per_member_loss

    ref, new, ref_mod = _build_models("spike")
    idx = torch.randint(0, VOCAB, (B, T))
    tgt = torch.randint(0, VOCAB, (B, T))
    noise = _sample_shared_noise(ref.P)
    coeff = 0.05 / (POP * SIGMA)
    rank_scale = 1.0 / math.sqrt(RANK)

    # reference path: inline update loop copied from Stage_0.train
    loss_r = ref_mod.per_member_loss(ref(idx, noise, SIGMA), tgt)
    fit_r = ref_mod.fitness_from_loss(loss_r).to(DTYPE)
    for name, par in ref.P.items():
        if name.endswith("_w"):
            A, Bn = noise[name]
            upd = rank_scale * torch.einsum("p,por,pir->oi", fit_r, A, Bn)
        else:
            upd = torch.tensordot(fit_r, noise[name], dims=([0], [0]))
        par.data.add_(coeff * upd.to(par.dtype))

    # modular path
    loss_n = per_member_loss(new(idx, noise, SIGMA), tgt)
    fit_n = fitness_from_loss(loss_n).to(DTYPE)
    es_update(new.P, noise, fit_n, coeff, rank_scale)

    max_err = max((ref.P[n] - new.P[n]).abs().max().item() for n in ref.P.keys())
    assert max_err < 1e-10, f"post-update param mismatch: max abs err {max_err}"
    print(f"post-update param max abs err: {max_err:.2e}")


def test_ternary_path_materialized_reference():
    """Guardrail #2: the Stage-1b ternary linear, with quantization DISABLED, must be
    numerically identical to the base eggroll_linear (it just materializes the perturbed
    weight instead of using base + low-rank correction). This pins the materialization
    math so the only behavioural change from ternary is the quantize() itself."""
    from spiking_m2rnn.eggroll import eggroll_linear, eggroll_linear_ternary

    P, O, I, Bn, S = 5, 8, 6, 3, 4
    torch.manual_seed(7)
    x = torch.randn(P, Bn, S, I, dtype=DTYPE)
    W = torch.randn(O, I, dtype=DTYPE)
    A = torch.randn(P, O, RANK, dtype=DTYPE)
    Bf = torch.randn(P, I, RANK, dtype=DTYPE)
    bias = torch.randn(O, dtype=DTYPE)
    bias_noise = torch.randn(P, O, dtype=DTYPE)
    rs = 1.0 / math.sqrt(RANK)

    base = eggroll_linear(x, W, A, Bf, bias=bias, bias_noise=bias_noise, sigma=SIGMA, rank_scale=rs)
    tern = eggroll_linear_ternary(x, W, A, Bf, bias=bias, bias_noise=bias_noise,
                                  sigma=SIGMA, rank_scale=rs, quantize=False)
    err = (base - tern).abs().max().item()
    assert err < 1e-10, f"ternary (no-quant) vs eggroll_linear mismatch: {err}"
    print(f"ternary materialized-reference max abs err: {err:.2e}")


if __name__ == "__main__":
    test_forward_spike()
    test_forward_tanh()
    test_es_update_matches()
    test_ternary_path_materialized_reference()
    print("OK: modular path is numerically identical to Stage_0 reference.")
