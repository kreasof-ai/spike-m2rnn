"""
Interleaved ES + GD trainer for the hybrid ternary/low-rank ViT (block coordinate descent).

Each iteration alternates two blocks on the SAME batch:
  1. ES block  -- perturb + update the ternary base (residual frozen). Population forward.
  2. GD block  -- exact-gradient update on the FP residual U,V (base frozen). Single forward.

The GD optimizer is STATELESS by design: ES moves the base underneath the residual between
steps, so any Adam-style moment buffer is computed against a stale base/curvature. Options:
rmsnorm (per-tensor RMS-normalized SGD -- one knob, the default), sign (signSGD), muon
(Newton-Schulz orthogonalized, momentum-free).

Brackets (set via flags) so the hybrid has controls to beat:
  --rank 0                  pure-ES ternary           (no residual, no GD)
  --no-ternary              pure-ES FP                (the original EGGROLL ViT regime)
  --rank 0 --no-ternary     pure-ES FP, plainest ES baseline
  --pure-gd                 standard full backprop FP (GD upper bound; ES off)

Run:  python hybrid/train.py --dataset mnist
      python hybrid/train.py --dataset mnist --rank 0          # pure-ES ternary control
      python hybrid/train.py --dataset cifar10 --steps 4000
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from hybrid_vit import (
    HybridViT, per_member_loss, fitness_from_loss, es_update, ternary_quantize,
)


# ============================ stateless GD step ============================
def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.t()
    for _ in range(steps):
        Z = X @ X.t()
        X = a * X + (b * Z + c * (Z @ Z)) @ X
    return X.t() if transposed else X


@torch.no_grad()
def stateless_gd_step(params, lr, kind):
    """In-place stateless update on the residual factors from their .grad."""
    for par in params:
        g = par.grad
        if g is None:
            continue
        if kind == "sign":
            step = lr * g.sign()
        elif kind == "muon" and g.dim() == 2:
            step = lr * (max(g.shape) ** 0.5) * newton_schulz(g).to(g.dtype)
        else:  # rmsnorm (and muon fallback for non-2D)
            gf = g.float()
            step = lr * (gf / (gf.norm() + 1e-8) * (gf.numel() ** 0.5)).to(g.dtype)
        par.data.add_(-step)
        par.grad = None


# ============================ data ============================
def get_loaders(dataset, batch_size):
    if dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        tr = datasets.MNIST("./data", train=True, download=True, transform=tfm)
        te = datasets.MNIST("./data", train=False, download=True, transform=tfm)
        meta = dict(image_size=28, patch_size=7, channels=1, num_classes=10)
    elif dataset == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        tr = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
        te = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
        meta = dict(image_size=32, patch_size=8, channels=3, num_classes=10)
    else:
        raise ValueError(dataset)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(te, batch_size=512, shuffle=False), meta)


def inf_batches(loader, device, dtype):
    while True:
        for data, target in loader:
            yield data.to(device, dtype), target.to(device)


# ============================ eval ============================
@torch.no_grad()
def evaluate(model, te, device, dtype, rank, ablate=False):
    """Accuracy on the test set using base (mean) weights. ablate=True zeros the residual,
    measuring how much the TERNARY BASE carries on its own (load-bearing check)."""
    saved = None
    if ablate and rank:
        saved = {k: v.data.clone() for k, v in model.R.items()}
        for v in model.R.values():
            v.data.zero_()
    correct = tot = 0
    for data, target in te:
        data, target = data.to(device, dtype), target.to(device)
        logits = model.gd_forward(data)
        correct += (logits.argmax(-1) == target).sum().item()
        tot += target.size(0)
    if saved is not None:
        for k, v in model.R.items():
            v.data.copy_(saved[k])
    return 100.0 * correct / tot


# ============================ pure-GD baseline ============================
def train_pure_gd(model, tr, te, args, device, dtype):
    """Standard full backprop on a FP ViT (ES off). The GD upper bound. Trains every base
    param (ternary forward optional via --no flip)."""
    model.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=3e-4, weight_decay=0.01)
    it = inf_batches(tr, device, dtype)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        data, target = next(it)
        logits = model.gd_forward(data)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            acc = evaluate(model, te, device, dtype, model.rank)
            print(f"[pure-gd] step {step:5d} | loss {loss.item():.4f} | val {acc:.2f}% "
                  f"| {time.time()-t0:.0f}s")


# ============================ hybrid trainer ============================
def train_hybrid(model, tr, te, args, device, dtype):
    rank_scale = 1.0 / math.sqrt(args.es_rank)
    coeff = args.lr / (args.pop * args.sigma)
    it = inf_batches(tr, device, dtype)
    t0 = time.time()
    es_evals = 0                                           # forward evals spent on ES (the cost metric)

    for step in range(1, args.steps + 1):
        data, target = next(it)

        # ---------------- ES block: update the ternary base ----------------
        noise = model.sample_noise(args.pop, args.es_rank, device, dtype)
        if args.chunk and args.chunk < args.pop:
            parts = []
            for s in range(0, args.pop, args.chunk):
                sub = {k: (tuple(t[s:s+args.chunk] for t in v) if isinstance(v, tuple)
                           else v[s:s+args.chunk]) for k, v in noise.items()}
                parts.append(per_member_loss(model.es_forward(data, sub, args.sigma, rank_scale), target))
            loss = torch.cat(parts)
        else:
            loss = per_member_loss(model.es_forward(data, noise, args.sigma, rank_scale), target)
        fit = fitness_from_loss(loss).to(dtype)
        es_update(model.P, noise, fit, coeff, rank_scale)
        es_evals += args.pop

        # ---------------- GD block: update the FP residual ----------------
        gd_loss = None
        if model.rank and not args.no_gd:
            for _ in range(args.gd_steps):
                logits = model.gd_forward(data)
                gd_loss = F.cross_entropy(logits, target)
                gd_loss.backward()
                stateless_gd_step(model.gd_params(), args.gd_lr, args.gd_opt)

        # ---------------- fold ----------------
        if args.fold_every and step % args.fold_every == 0:
            model.fold_residual()

        # ---------------- logging / eval ----------------
        if step % args.eval_every == 0 or step == 1:
            acc = evaluate(model, te, device, dtype, model.rank)
            extra = ""
            if model.rank and not args.no_gd:
                acc_abl = evaluate(model, te, device, dtype, model.rank, ablate=True)
                gd_s = f"{gd_loss.item():.4f}" if gd_loss is not None else "  -  "
                extra = f"| ternary-only {acc_abl:.2f}% | gd {gd_s} "
            print(f"step {step:5d} | es-min {loss.min().item():.4f} | es-mean {loss.mean().item():.4f} "
                  f"| val {acc:.2f}% {extra}| es-evals {es_evals/1e6:.1f}M | {time.time()-t0:.0f}s")


# ============================ adaptive hybrid trainer ============================
def train_adaptive(model, tr, te, args, device, dtype):
    """Self-regulating GD gate + POP ramp (block coordinate descent with a trust region).

    Each step, with probability p_gd, ATTEMPT a GD move: grow the residual by gradient, then
    FOLD it into the ternary base and KEEP the fold only if it lowers base-only loss on a
    held-out probe (i.e. the gradient-proposed ternary flips actually help); else roll the
    base back. The fold makes the proposal a concrete, testable base change -- and it
    self-anneals: once the base is good, the small rank-r residual no longer crosses any
    quantization boundary, folds become no-ops, and GD dies out on its own.

    Feedback:  reject -> p_gd *= down (harder to pick GD);  accept -> p_gd recovers slowly.
    POP ramp:  pop scales pop_min -> pop_max as p_gd falls (cheap while GD warms up, large-pop
    ES to sharpen the asymptote late). One signal drives the filter, the anneal, and the ramp.
    """
    rank_scale = 1.0 / math.sqrt(args.es_rank)
    it = inf_batches(tr, device, dtype)
    probe_it = inf_batches(tr, device, dtype)              # independent batches for the trust test
    p_gd = args.pgd_init
    t0 = time.time()
    es_evals = 0
    win_attempt = win_accept = 0                            # accept stats over the logging window

    def cur_pop():
        raw = args.pop_min + (args.pop_max - args.pop_min) * (1.0 - p_gd)
        step = max(args.chunk, 1)
        return max(args.pop_min, int(round(raw / step) * step))

    for step in range(1, args.steps + 1):
        data, target = next(it)
        pop = cur_pop()

        # ---------------- ES block (pop varies with the gate) ----------------
        coeff = args.lr / (pop * args.sigma)
        noise = model.sample_noise(pop, args.es_rank, device, dtype)
        if args.chunk and args.chunk < pop:
            parts = []
            for s in range(0, pop, args.chunk):
                sub = {k: (tuple(t[s:s+args.chunk] for t in v) if isinstance(v, tuple)
                           else v[s:s+args.chunk]) for k, v in noise.items()}
                parts.append(per_member_loss(model.es_forward(data, sub, args.sigma, rank_scale), target))
            loss = torch.cat(parts)
        else:
            loss = per_member_loss(model.es_forward(data, noise, args.sigma, rank_scale), target)
        fit = fitness_from_loss(loss).to(dtype)
        es_update(model.P, noise, fit, coeff, rank_scale)
        es_evals += pop

        # ---------------- gated GD block (trust-region fold) ----------------
        if model.rank and not args.no_gd and torch.rand(1).item() < p_gd:
            win_attempt += 1
            pdata, ptarget = next(probe_it)
            with torch.no_grad():
                l_before = F.cross_entropy(model.gd_forward(pdata, use_residual=False), ptarget).item()
            snap = model.snapshot_base()
            for _ in range(args.gd_steps):                 # grow residual from 0 by gradient
                gd_loss = F.cross_entropy(model.gd_forward(data), target)
                gd_loss.backward()
                stateless_gd_step(model.gd_params(), args.gd_lr, args.gd_opt)
            model.fold_residual()                           # commit proposal into the base, reset residual
            with torch.no_grad():
                l_after = F.cross_entropy(model.gd_forward(pdata, use_residual=False), ptarget).item()
            if l_after <= l_before - args.accept_margin:    # the flips helped -> keep
                p_gd = min(1.0, p_gd * args.pgd_up)
                win_accept += 1
            else:                                           # harmful/useless -> roll back, distrust GD
                model.restore_base(snap)
                p_gd *= args.pgd_down

        # ---------------- logging / eval ----------------
        if step % args.eval_every == 0 or step == 1:
            acc = evaluate(model, te, device, dtype, model.rank)   # residual ~0 (folded) so == base
            rate = win_accept / max(win_attempt, 1)
            print(f"step {step:5d} | es-min {loss.min().item():.4f} | val {acc:.2f}% "
                  f"| p_gd {p_gd:.3f} | pop {pop} | gd-accept {rate:.2f} ({win_accept}/{win_attempt}) "
                  f"| es-evals {es_evals/1e6:.1f}M | {time.time()-t0:.0f}s")
            win_attempt = win_accept = 0


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=50)
    # model
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp", type=int, default=128)
    ap.add_argument("--rank", type=int, default=4, help="FP residual rank (0 = pure ES, no GD)")
    ap.add_argument("--no-ternary", action="store_true", help="FP base (ES-FP control)")
    # ES
    ap.add_argument("--pop", type=int, default=1024)
    ap.add_argument("--es-rank", type=int, default=1, help="ES low-rank noise rank")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=512)
    # GD
    ap.add_argument("--gd-lr", type=float, default=0.02)
    ap.add_argument("--gd-opt", choices=["rmsnorm", "sign", "muon"], default="rmsnorm")
    ap.add_argument("--gd-steps", type=int, default=1)
    ap.add_argument("--no-gd", action="store_true", help="freeze residual (pure ES with dead scaffold)")
    ap.add_argument("--fold-every", type=int, default=0, help="fold residual into base every N steps (0=off)")
    # adaptive controller (self-regulating GD gate + POP ramp)
    ap.add_argument("--adaptive", action="store_true", help="gated trust-region GD with annealing + POP ramp")
    ap.add_argument("--pgd-init", type=float, default=1.0, help="initial P(attempt GD)")
    ap.add_argument("--pgd-down", type=float, default=0.9, help="multiply p_gd on a rejected GD fold")
    ap.add_argument("--pgd-up", type=float, default=1.02, help="multiply p_gd on an accepted GD fold")
    ap.add_argument("--accept-margin", type=float, default=0.0, help="min base-loss drop to accept a fold")
    ap.add_argument("--pop-min", type=int, default=512, help="POP when GD is fully on (p_gd=1)")
    ap.add_argument("--pop-max", type=int, default=4096, help="POP when GD is off (p_gd=0)")
    # modes
    ap.add_argument("--pure-gd", action="store_true", help="standard full backprop baseline (ES off)")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[args.dtype]

    tr, te, meta = get_loaders(args.dataset, args.batch_size)
    model = HybridViT(dim=args.dim, depth=args.depth, heads=args.heads, mlp=args.mlp,
                      rank=args.rank, ternary=not args.no_ternary, **meta).to(device).to(dtype)

    n_tern = sum(p.numel() for n, p in model.P.items() if n.endswith("_w"))
    n_res = sum(p.numel() for p in model.R.values())
    print(f"hybrid ViT | {args.dataset} | dim={args.dim} depth={args.depth} | "
          f"ternary={not args.no_ternary} rank={args.rank} | "
          f"base(ternary) params={n_tern/1e3:.0f}k  residual(fp) params={n_res/1e3:.1f}k "
          f"({100*n_res/max(n_tern,1):.1f}% of base) | gd-opt={args.gd_opt} | dev={device}")

    if args.pure_gd:
        train_pure_gd(model, tr, te, args, device, dtype)
    else:
        model.requires_grad_(False)
        for p in model.gd_params():                        # only the residual needs grad
            p.requires_grad_(True)
        if args.adaptive:
            print(f"adaptive | p_gd init={args.pgd_init} down={args.pgd_down} up={args.pgd_up} "
                  f"| pop {args.pop_min}->{args.pop_max} | margin={args.accept_margin}")
            train_adaptive(model, tr, te, args, device, dtype)
        else:
            train_hybrid(model, tr, te, args, device, dtype)


if __name__ == "__main__":
    main()
