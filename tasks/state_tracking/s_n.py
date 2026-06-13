"""
S_n state-tracking task (the group word problem) -- the decisive Stage-0.5b
experiment's data generator.

The task: given a sequence of permutations g_1, g_2, ..., g_T drawn from the
symmetric group S_n, predict at every step t the cumulative composition
    P_t = g_1 . g_2 . ... . g_t    (left-to-right function composition).
Tracking P_t requires carrying the FULL group element forward -- a Transformer /
diagonal-SSM (TC0) cannot do it for non-solvable groups, while a finite-precision
non-linear RNN (FSA, NC1) can. S5 is non-solvable => its word problem is
NC1-complete; S3 is solvable but still a clean state-tracking warm-up.

Length generalization is the whole point: TRAIN at short T, EVAL at longer T. A
real FSA holds accuracy flat as T grows; a model that "cheats" with positional
shortcuts collapses. Reproduce M2RNN's perfect-generalization plot.

Encoding (matches `data.get_batch`'s contract -- (B, T) long tensors):
  * vocab = n!  (every permutation of {0..n-1} is one token, indexed by `perm_index`)
  * x[b, t] = token id of generator g_t
  * y[b, t] = token id of the cumulative product P_t
So the model maps a stream of permutation tokens to the running-product token.

Use `generators="all"` (default) for the full group, or pass a small generating
set (e.g. an adjacent transposition + an n-cycle) to make the task harder /
more FSA-like. Either way every reachable state is a full S_n element.
"""

import itertools
import math

import torch


class SymmetricGroup:
    """All permutations of {0..n-1}, with composition + indexing helpers.

    Permutations are stored as tuples p where p[i] is the image of i. Composition
    is left-to-right: (a then b)[i] = b[a[i]], so the cumulative product of the
    input stream reads naturally in sequence order.
    """

    def __init__(self, n):
        self.n = n
        self.perms = [tuple(p) for p in itertools.permutations(range(n))]   # n! of them
        self.index = {p: i for i, p in enumerate(self.perms)}
        self.size = len(self.perms)                                         # vocab
        self.identity = tuple(range(n))
        # precompute the Cayley table: compose[i][j] = index of (perm_i then perm_j)
        self.compose_table = torch.empty(self.size, self.size, dtype=torch.long)
        for i, a in enumerate(self.perms):
            for j, b in enumerate(self.perms):
                self.compose_table[i, j] = self.index[tuple(b[a[k]] for k in range(n))]

    def perm_index(self, p):
        return self.index[tuple(p)]

    def default_generators(self):
        """A 2-element generating set of S_n: adjacent transposition (0 1) and the
        n-cycle (0 1 ... n-1). For n>=2 these generate the whole group."""
        swap = list(range(self.n)); swap[0], swap[1] = 1, 0
        cyc  = [(i + 1) % self.n for i in range(self.n)]
        return [self.index[tuple(swap)], self.index[tuple(cyc)]]


def make_batch(group, batch, length, device, generators="all", seed=None):
    """Return (x, y) long tensors of shape (batch, length).

    x[b, t] = token id of the t-th input generator
    y[b, t] = token id of the cumulative product g_1 . ... . g_t
    `generators`: "all" => sample uniformly from the whole group; or a list of
    token ids to sample the input stream from (the reachable states are still the
    full group).
    """
    # IMPORTANT: a fresh torch.Generator() has a FIXED default seed, so creating one
    # per call (unseeded) returns the SAME data every call -- which silently kills all
    # averaging in eval AND makes training see one fixed batch per length. So only use
    # an explicit generator when `seed` is given (reproducibility); otherwise fall back
    # to the global RNG (generator=None), which advances on every call.
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

    if generators == "all":
        pool = torch.arange(group.size)
    else:
        pool = torch.as_tensor(list(generators), dtype=torch.long)

    pick = torch.randint(len(pool), (batch, length), generator=g)
    x = pool[pick]                                                  # (B, T) input tokens

    table = group.compose_table                                    # (|G|, |G|) on cpu
    y = torch.empty_like(x)
    cur = torch.full((batch,), group.perm_index(group.identity), dtype=torch.long)
    for t in range(length):
        cur = table[cur, x[:, t]]                                  # P_t = P_{t-1} . g_t
        y[:, t] = cur
    return x.to(device), y.to(device)


def get_batch_fn(group, generators="all"):
    """Adapter matching the (data, batch, block, device) signature of
    `data.get_batch`, so the train loop can stay task-agnostic. `data` is ignored
    (the task is generated on the fly)."""
    def get_batch(data, batch, block, device):
        return make_batch(group, batch, block, device, generators=generators)
    return get_batch


if __name__ == "__main__":
    # self-check: cumulative product is correct and associative under the table
    for n in (3, 5):
        G = SymmetricGroup(n)
        x, y = make_batch(G, batch=4, length=20, device=torch.device("cpu"), seed=0)
        # recompute y by brute force from perms and compare
        ok = True
        for b in range(x.shape[0]):
            cur = G.identity
            for t in range(x.shape[1]):
                gp = G.perms[x[b, t].item()]
                cur = tuple(gp[cur[k]] for k in range(n))
                if G.perm_index(cur) != y[b, t].item():
                    ok = False
        print(f"S{n}: vocab={G.size}, gens={G.default_generators()}, batch ok={ok}")
    print("OK: S_n cumulative-product generator self-check passed.")
