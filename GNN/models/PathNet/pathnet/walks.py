"""Random-walk path sampling for PathNet (MERW, the upstream default, or plain RW).

Upstream PathNet samples, for every node, ``num_walks`` walks of length
``walk_length`` using a Maximal-Entropy Random Walk (MERW) whose transition
probabilities come from the Perron eigenvector of the (symmetric) adjacency
matrix. It also stores, per walked node, its graph distance to the start node
(the ego is ``0``), used to pick a distance-specific linear map.

This module reproduces that on the MAGNN-derived IMDB / Freebase graphs:

* ``merw_transition`` builds the MERW transition rows ``P_ij = A_ij psi_j /
  (lambda psi_i)`` (``rw`` falls back to the degree-normalised walk).
* ``sample_walks`` draws walks starting only from the *target* (labelled) nodes
  — the only rows the model ever consumes — which keeps sampling cheap on the
  large Freebase graph.
* distance-to-target is the earliest walk position at which a node is reached
  across that target's walks (a deterministic proxy for BFS graph distance that
  correctly collapses revisits to their nearest layer, capped at ``L-1``).

Walk tensors are cached to disk keyed by every sampling knob so repeated runs /
seeds are reproducible and fast.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class WalkData:
    """Walk tensors for a single graph + target set."""

    walks: torch.Tensor      # [T, W, L] long node ids (walks[:, :, 0] == target)
    dists: torch.Tensor      # [T, W, L] long distance-to-target buckets in [0, L-1]
    target_nodes: torch.Tensor  # [T] long global node ids
    num_nodes: int
    walk_type: str

    def to(self, device) -> "WalkData":
        return WalkData(
            walks=self.walks.to(device),
            dists=self.dists.to(device),
            target_nodes=self.target_nodes.to(device),
            num_nodes=self.num_nodes,
            walk_type=self.walk_type,
        )


def build_symmetric_adj(edges: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    """0/1 symmetric adjacency (no self loops) from an ``[E, 2]`` edge array."""
    if edges.size == 0:
        raise ValueError("Empty edge set; cannot build adjacency.")
    r = edges[:, 0].astype(np.int64)
    c = edges[:, 1].astype(np.int64)
    data = np.ones(r.shape[0], dtype=np.float64)
    A = sp.csr_matrix((data, (r, c)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)            # symmetric
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A


def _perron_vector(A: sp.csr_matrix) -> np.ndarray:
    """Dominant (Perron) eigenvector of the symmetric adjacency, strictly > 0."""
    n = A.shape[0]
    # ARPACK's unconstrained eigsh call can spend hours on this large,
    # disconnected graph before emitting any progress.  Fixed, converged power
    # iteration is deterministic and gives the Perron weights MERW needs.
    rng = np.random.RandomState(0)
    psi = rng.rand(n) + 0.1
    psi /= np.linalg.norm(psi)
    for iteration in range(1, 201):
        nxt = A.dot(psi)
        nrm = np.linalg.norm(nxt)
        if nrm == 0:
            break
        nxt /= nrm
        delta = np.linalg.norm(nxt - psi)
        psi = nxt
        if iteration % 25 == 0:
            print(
                f"  [walks] MERW power iteration {iteration}/200 "
                f"(delta={delta:.3e})",
                flush=True,
            )
        if delta < 1e-7:
            print(f"  [walks] MERW converged at iteration {iteration}", flush=True)
            break
    # Isolated nodes / numerical zeros: floor so division stays finite.
    psi = np.abs(psi)
    psi[psi < 1e-12] = 1e-12
    return psi


def build_transition_rows(A: sp.csr_matrix, walk_type: str):
    """Per-node ``(neighbor_ids, probabilities)`` lists for the chosen walk."""
    n = A.shape[0]
    indptr, indices, _ = A.indptr, A.indices, A.data
    if walk_type == "merw":
        # P_ij = A_ij psi_j / (lambda psi_i); the lambda / psi_i factors are
        # constant within a row, so per-row normalisation by psi_j suffices.
        weights = _perron_vector(A)
    elif walk_type == "rw":
        weights = None
    else:
        raise ValueError(f"Unknown walk_type {walk_type!r}; use 'merw' or 'rw'.")

    nbrs, probs = [], []
    for u in range(n):
        s, e = indptr[u], indptr[u + 1]
        nb = indices[s:e]
        if nb.size == 0:
            nbrs.append(nb.astype(np.int64))
            probs.append(np.zeros(0, dtype=np.float64))
            continue
        if weights is None:
            p = np.full(nb.size, 1.0 / nb.size, dtype=np.float64)
        else:
            w = weights[nb]
            tot = w.sum()
            p = (w / tot) if tot > 0 else np.full(nb.size, 1.0 / nb.size)
        nbrs.append(nb.astype(np.int64))
        probs.append(p)
    return nbrs, probs


def sample_walks(
    target_nodes: np.ndarray,
    nbrs: list,
    probs: list,
    num_walks: int,
    walk_length: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample walks for the target nodes.

    Returns ``walks`` and ``dists`` arrays of shape ``[T, W, L]``.
    A node with no neighbours stays put (self-loop) so the walk length is fixed.
    """
    T = int(target_nodes.shape[0])
    W, L = int(num_walks), int(walk_length)
    walks = np.empty((T, W, L), dtype=np.int64)
    walks[:, :, 0] = target_nodes[:, None]

    cur = np.repeat(target_nodes, W).astype(np.int64)  # [T*W]
    for step in range(1, L):
        nxt = cur.copy()
        # Group identical current nodes so np.random.choice is vectorised per node.
        order = np.argsort(cur, kind="stable")
        cs = cur[order]
        boundaries = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        for bi, start in enumerate(boundaries):
            end = boundaries[bi + 1] if bi + 1 < len(boundaries) else len(cs)
            node = int(cs[start])
            nb = nbrs[node]
            count = end - start
            if nb.size == 0:
                continue  # dead end: keep current node (self-loop)
            picks = rng.choice(nb, size=count, p=probs[node])
            nxt[order[start:end]] = picks
        walks[:, :, step] = nxt.reshape(T, W)
        cur = nxt

    # distance-to-target = earliest walk position a node is reached (per target),
    # collapsing revisits to their nearest layer; capped at L-1.
    dists = np.empty((T, W, L), dtype=np.int64)
    pos = np.arange(L, dtype=np.int64)
    for t in range(T):
        wt = walks[t]                 # [W, L]
        first: dict[int, int] = {}
        for w in range(W):
            for l in range(L):
                node = int(wt[w, l])
                if node not in first or l < first[node]:
                    first[node] = l
        dt = dists[t]
        for w in range(W):
            for l in range(L):
                dt[w, l] = min(first[int(wt[w, l])], L - 1)
    return walks, dists


def _cache_key(**kwargs) -> str:
    blob = "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def build_walk_data(
    edges: np.ndarray,
    num_nodes: int,
    target_nodes: np.ndarray,
    *,
    num_walks: int = 40,
    walk_length: int = 4,
    walk_type: str = "merw",
    seed: int = 0,
    cache_dir: str | None = None,
    cache_tag: str = "",
    use_cache: bool = True,
) -> WalkData:
    """Build (and cache) walk tensors for a graph + target-node set."""
    target_nodes = np.asarray(target_nodes, dtype=np.int64)
    cache_file = None
    if cache_dir and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(
            tag=cache_tag,
            n=num_nodes,
            t=int(target_nodes.shape[0]),
            e=int(edges.shape[0]),
            nw=num_walks,
            wl=walk_length,
            wt=walk_type,
            seed=seed,
        )
        cache_file = os.path.join(cache_dir, f"walks_{cache_tag}_{key}.pt")
        if os.path.isfile(cache_file):
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            return payload

    A = build_symmetric_adj(edges, num_nodes)
    print(
        f"  [walks] building {walk_type} transitions on {num_nodes} nodes / "
        f"{edges.shape[0]} edges...",
        flush=True,
    )
    nbrs, probs = build_transition_rows(A, walk_type)
    print(
        f"  [walks] sampling {target_nodes.shape[0]} x {num_walks} walks "
        f"of length {walk_length}...",
        flush=True,
    )
    rng = np.random.RandomState(seed)
    walks, dists = sample_walks(target_nodes, nbrs, probs, num_walks, walk_length, rng)
    print("  [walks] sampling done", flush=True)

    data = WalkData(
        walks=torch.from_numpy(walks).long(),
        dists=torch.from_numpy(dists).long(),
        target_nodes=torch.from_numpy(target_nodes).long(),
        num_nodes=int(num_nodes),
        walk_type=walk_type,
    )
    if cache_file is not None:
        torch.save(data, cache_file)
    return data
