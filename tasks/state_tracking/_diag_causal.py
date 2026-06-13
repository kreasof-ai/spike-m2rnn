"""Diagnostic: is the forward causal? logits at positions 0..t-1 must NOT depend
on total sequence length T. Compares a full length-L run against a truncated run
on the same prefix. Eager (no compile), CPU, float64."""
import os, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn.model import SpikingM2RNN
from s_n import SymmetricGroup, make_batch

torch.manual_seed(0)
G = SymmetricGroup(5)
dev = torch.device("cpu")
for mode in ("tanh", "spike"):
    model = SpikingM2RNN(G.size, dim=32, depth=2, k=16, v=16, mlp=32, mode=mode,
                         threshold=1.0, decay=1.0).to(dev).double()
    model.eval(); model.requires_grad_(False)
    zn = model.zero_noise(dev, torch.float64)

    x_long, _ = make_batch(G, batch=3, length=64, device=dev, seed=1)
    short = 16
    x_short = x_long[:, :short]

    lo_long = model(x_long, zn, 0.0)[0]      # (B,64,V)
    lo_short = model(x_short, zn, 0.0)[0]    # (B,16,V)
    err = (lo_long[:, :short] - lo_short).abs().max().item()
    print(f"[{mode}] max |logit(prefix of L64) - logit(L16)| over positions 0..15 = {err:.3e}"
          f"  -> {'CAUSAL ok' if err < 1e-9 else 'NON-CAUSAL / length-dependent BUG'}")
