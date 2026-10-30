"""Exact distance-to-target buckets for PathNet walks — upstream's definition.

Upstream PathNet (``preprocess/gen_merw.cpp``) writes, for every node of a sampled walk, ``dis[start][node]``: the BFS
shortest-path distance from the walk's start node in the graph (precomputed per start, ≤ ``seq_len`` hops; revisits get the
same value). The port's sampler (``pathnet/walks.py``) used a proxy instead — the earliest walk position at
which the node appears among that target's walks — which can over-estimate the distance (e.g. a node first reached at step
3 that is also a 1-hop neighbour of the target).

On a symmetric graph a node at walk position j has BFS distance ≤ j ≤ L − 1 = 3, so the exact value needs only:
  0 if node == target; 1 if node ∈ N(target); 2 if N(node) ∩ N(target) ≠ ∅ (or the node sits at position 2); else 3.
The adjacency is the arm's own walk graph (physical / universal / compiled program) — identical across variants on the
invariant arm, so the buckets are too.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def bfs_distance_buckets(walks: np.ndarray, A: sp.csr_matrix) -> np.ndarray:
    """walks [T, W, L] global node ids (walks[:, :, 0] = target); A symmetric 0/1 CSR. Returns dists [T, W, L] int64."""
    A = A.tocsr(); A.sort_indices()
    indptr, indices = A.indptr, A.indices
    T, W, L = walks.shape
    dists = np.empty_like(walks, dtype=np.int64)
    pos = np.broadcast_to(np.arange(L), (W, L))
    for i in range(T):
        t = int(walks[i, 0, 0]); w = walks[i]
        nt = indices[indptr[t]:indptr[t + 1]]
        d = np.minimum(pos, L - 1).astype(np.int64).copy()          # upper bound: BFS distance <= walk position
        d[w == t] = 0
        is_nbr = np.isin(w, nt) & (w != t)
        d[is_nbr] = 1
        todo = np.unique(w[(d >= 3)])                                 # position-3 nodes not yet resolved
        two = set()
        for v in todo.tolist():
            nv = indices[indptr[v]:indptr[v + 1]]
            small, big = (nt, nv) if nt.size <= nv.size else (nv, nt)
            if small.size and big.size:
                k = np.searchsorted(big, small)
                k[k >= big.size] = big.size - 1
                if np.any(big[k] == small):
                    two.add(v)
        if two:
            d[np.isin(w, np.fromiter(two, dtype=w.dtype)) & (d >= 3)] = 2
        dists[i] = d
    return dists
