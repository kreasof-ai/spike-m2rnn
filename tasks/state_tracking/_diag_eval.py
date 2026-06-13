"""Diagnostic: does per-position accuracy depend on TOTAL sequence length?
For a causal model it must not. Train a small S3 model briefly, then measure
acc-per-position at L=16 and L=64 with many samples and compare positions 0..15.
Also checks whether make_batch's unseeded Generator actually randomizes per call."""
import os, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn import config
from spiking_m2rnn.eggroll import es_update, fitness_from_loss, per_member_loss
from spiking_m2rnn.model import SpikingM2RNN
from s_n import SymmetricGroup, make_batch

dev = torch.device("cpu")
dt = torch.float32

# --- check 0: does an unseeded make_batch randomize per call? ---
G = SymmetricGroup(3)
a, _ = make_batch(G, 4, 8, dev)
b, _ = make_batch(G, 4, 8, dev)
print(f"[gen] two unseeded make_batch identical? {torch.equal(a, b)}  "
      f"(True => Generator() NOT randomized => eval has no averaging)")

# --- train a small S3 model so accuracy is above chance ---
cfg = config.Config(device=dev, dtype=dt, mode="tanh", dim=32, depth=1, k_dim=16,
                    v_dim=16, mlp_dim=32, pop_size=128, batch_size=32, sigma=0.05, decay=1.0)
model = SpikingM2RNN(G.size, dim=cfg.dim, depth=cfg.depth, k=cfg.k_dim, v=cfg.v_dim,
                     mlp=cfg.mlp_dim, mode=cfg.mode, threshold=cfg.threshold, decay=cfg.decay)
model.to(dev).to(dt); model.eval(); model.requires_grad_(False)
for step in range(400):
    tl = [8, 16][step % 2]
    x, y = make_batch(G, cfg.batch_size, tl, dev)
    noise = model.sample_noise(cfg.pop_size, cfg.rank, dev, dt)
    loss = per_member_loss(model(x, noise, cfg.sigma), y)
    fit = fitness_from_loss(loss).to(dt)
    es_update(model.P, noise, fit, cfg.coeff, cfg.rank_scale)

@torch.no_grad()
def acc_per_pos(length, batches=300):
    zn = model.zero_noise(dev, dt)
    correct = torch.zeros(length, dtype=torch.float64)
    n = 0
    for _ in range(batches):
        x, y = make_batch(G, cfg.batch_size, length, dev)
        pred = model(x, zn, 0.0)[0].argmax(-1)
        correct += (pred == y).float().sum(0).double()
        n += x.shape[0]
    return correct / n

p16 = acc_per_pos(16)
p64 = acc_per_pos(64)
print(f"\nacc at positions 0..15, measured in an L16 eval vs an L64 eval:")
print("  L16:", " ".join(f"{v*100:4.1f}" for v in p16.tolist()))
print("  L64:", " ".join(f"{v*100:4.1f}" for v in p64[:16].tolist()))
diff = (p16 - p64[:16]).abs().max().item()
print(f"\nmean over pos 0..15:  L16={p16.mean()*100:.2f}%   L64={p64[:16].mean()*100:.2f}%")
print(f"max |diff| at pos 0..15 between the two lengths = {diff*100:.2f} pts")
print("-> " + ("CONSISTENT (per-position acc is length-independent, as causality requires)"
               if diff < 0.03 else "INCONSISTENT (per-position acc depends on total length — real bug)"))
