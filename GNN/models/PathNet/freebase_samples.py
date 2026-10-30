"""RGCN's deterministic Freebase neighbour sampler, vectorised; PathNet uses its store table (``STORES``).

RGCN's Freebase runner (``RGCN/freebase_conditional_rgcn.py::build_two_layer_blocks``) samples, for a batch of
target books, ``fanout`` incoming neighbours per (destination, channel): layer 2 into the targets, layer 1 into the
layer-2 closure. Positions come from ``deterministic_positions`` keyed by splitmix64 of (sampling_seed, layer,
store channel id, destination) — a function of the semantic store, never of batch composition or memory layout.
Training re-samples every step with ``splitmix_seed(seed, epoch, batch_no)``; evaluation uses fixed seeds
(val ``seed + 1_000_000 + batch_no``; final train/val/test ``seed + 2_000_000 + split * 100_000 + batch_no``).

``sample_two_layers`` reproduces that edge set exactly (``verify_against_rgcn`` checks it against the scalar RGCN
functions, imported read-only) with numpy uint64 arithmetic per channel instead of a Python loop per
(destination, channel). The batch MESSAGE GRAPH for an attention-diffusion model is the union of the two layers'
edges (set semantics) over the layer-1 closure, typed by a global (semantic_relation, direction) slot so every arm
shares one relation table (RGCN's ``relation_vocabulary`` rule for augmentation). On the invariant arm the three
variants' stores are hash-identical (``combined_model_input_hash``), so every sample and every step is identical.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "RGCN"))
from freebase_conditional_rgcn import BlockConditionalGraph  # noqa: E402  (RGCN, read-only)

PRE = Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parent / "data"))) / "preprocessed"
STORES = {
    ("physical", "unchanged"): PRE / "freebase_rgcn_physical/unchanged/physical_unchanged/union/all_graph",
    ("physical", "exact_2"): PRE / "freebase_rgcn_physical/exact_2/physical_exact_2/union/all_graph",
    ("physical", "exact_3"): PRE / "freebase_rgcn_physical/exact_3/physical_exact_3/union/all_graph",
    ("physical", "union_exact_2_3"): PRE / "freebase_rgcn_physical/union_exact_2_3/physical_union_exact_2_3/union/all_graph",
    ("semantic", "unchanged"): PRE / "freebase_rgcn_block_conditional/unchanged/exact_3/union/all_graph",
    ("semantic", "exact_2"): PRE / "freebase_rgcn_block_conditional/exact_2/exact_3/union/all_graph",
    ("semantic", "exact_3"): PRE / "freebase_rgcn_block_conditional/exact_3/exact_3/union/all_graph",
}
U64 = np.uint64
C_LAYER, C_CHAN, C_DST, C_STEP = U64(0x9E3779B97F4A7C15), U64(0xBF58476D1CE4E5B9), U64(0x94D049BB133111EB), U64(0xD6E8FEB86659FD93)


def splitmix64_vec(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = z + U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def splitmix_seed(seed: int, epoch: int, batch: int) -> int:
    """= RGCN/run_Freebase_rgcn_block_conditional.py::splitmix_seed (per-step training sample seed)."""
    value = (int(seed) ^ (int(epoch) << 21) ^ (int(batch) << 42)) & ((1 << 63) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 63) - 1)
    value ^= value >> 27
    return int(value)


def global_vocabulary() -> dict[tuple[str, int], int]:
    keys = set()
    for root in STORES.values():
        for rec in json.load(open(root / "relation_catalog.json"))["channels"]:
            keys.add((str(rec["semantic_relation"]), int(rec["direction"])))
    return {k: i for i, k in enumerate(sorted(keys))}


class Store:
    def __init__(self, graph_kind: str, variant: str, vocab: dict):
        self.root = STORES[(graph_kind, variant)]
        self.g = BlockConditionalGraph(self.root, mmap=True)          # no vocabulary: channel_id = store id (RGCN's key)
        self.hashes = json.load(open(self.root / "model_input_hashes.json"))
        self.slot = np.array([vocab[(c.name, c.direction)] for c in self.g.channels], dtype=np.int64)
        self.node_types = np.asarray(self.g.node_types, dtype=np.int64)
        self.targets = np.asarray(self.g.target_nodes, dtype=np.int64)
        self.labels = np.asarray(self.g.labels, dtype=np.int64)
        self.splits = self.g.splits
        self.num_nodes = self.g.num_nodes

    def sample_layer(self, dst_nodes: np.ndarray, fanout: int, seed: int, layer: int) -> np.ndarray:
        """(src, dst, store_channel) of build_sampled_block(dst_nodes, fanout, seed, layer), any order."""
        out = []
        dtypes = self.node_types[dst_nodes]
        with np.errstate(over="ignore"):
            base = U64(int(seed) & 0xFFFFFFFFFFFFFFFF) ^ (U64(int(layer)) * C_LAYER)
        for ch in self.g.channels:
            d = dst_nodes[dtypes == ch.destination_type]
            if not d.size:
                continue
            lo = np.asarray(ch.indptr[d], dtype=np.int64); deg = np.asarray(ch.indptr[d + 1], dtype=np.int64) - lo
            keep = deg > 0
            d, lo, deg = d[keep], lo[keep], deg[keep]
            if not d.size:
                continue
            small = deg <= fanout if fanout > 0 else np.ones_like(deg, dtype=bool)
            if small.any():
                ds, ls, gs = d[small], lo[small], deg[small]
                pos = np.repeat(ls - np.concatenate([[0], np.cumsum(gs)[:-1]]), gs) + np.arange(gs.sum())
                out.append(np.stack([np.asarray(ch.indices[pos], dtype=np.int64), np.repeat(ds, gs), np.full(pos.size, ch.channel_id)], 1))
            big = ~small
            if big.any():
                db, lb, gb = d[big], lo[big], deg[big]
                with np.errstate(over="ignore"):
                    key = base ^ (U64(int(ch.channel_id)) * C_CHAN) ^ (db.astype(U64) * C_DST)
                offset = (splitmix64_vec(key) % gb.astype(U64)).astype(np.int64)
                step = (U64(1) + splitmix64_vec(key ^ C_STEP) % (gb - 1).astype(U64)).astype(np.int64)
                bad = np.gcd(step, gb) != 1
                while bad.any():                                        # RGCN: step += 1; wrap to 1 at degree
                    step[bad] += 1
                    step[bad & (step >= gb)] = 1
                    bad = np.gcd(step, gb) != 1
                pos = (offset[:, None] + np.arange(fanout)[None, :] * step[:, None]) % gb[:, None]
                src = np.asarray(ch.indices[(lb[:, None] + pos).ravel()], dtype=np.int64)
                out.append(np.stack([src, np.repeat(db, fanout), np.full(src.size, ch.channel_id)], 1))
        return np.concatenate(out) if out else np.empty((0, 3), np.int64)

    def sample_two_layers(self, targets: np.ndarray, fanout1: int, fanout2: int, seed: int):
        """-> nodes (sorted global ids = RGCN block1.src_nodes), local message edges (src, dst, slot) unique, target rows."""
        targets = np.asarray(targets, dtype=np.int64)
        e2 = self.sample_layer(targets, fanout2, seed, 2)
        mid = np.unique(np.concatenate([targets, e2[:, 0]]))             # = block2.src_nodes = block1.dst_nodes
        e1 = self.sample_layer(mid, fanout1, seed, 1)
        E = np.unique(np.concatenate([e2, e1]), axis=0) if len(e2) + len(e1) else np.empty((0, 3), np.int64)
        nodes = np.unique(np.concatenate([mid, E[:, 0]]))
        loc = np.searchsorted(nodes, E[:, :2])
        return nodes, np.stack([loc[:, 0], loc[:, 1], self.slot[E[:, 2]]], 1), np.searchsorted(nodes, targets)


def verify_against_rgcn(store: Store, n_batches: int = 3, batch: int = 64, fanout: int = 4) -> bool:
    """Edge sets of the vectorised sampler == RGCN's build_two_layer_blocks, on real batches."""
    from freebase_conditional_rgcn import build_two_layer_blocks
    rng = np.random.default_rng(0)
    ok = True
    for b in range(n_batches):
        tg = np.sort(rng.choice(store.targets, batch, replace=False))
        seed = splitmix_seed(1566911444, 1, b)
        b1, b2 = build_two_layer_blocks(store.g, tg, fanout_layer1=fanout, fanout_layer2=fanout, sampling_seed=seed)
        ref = set()
        for blk in (b1, b2):
            s = blk.src_nodes[blk.src_index.numpy()]; d = blk.dst_nodes[blk.dst_index.numpy()]
            ref |= set(zip(s.tolist(), d.tolist(), blk.edge_type.numpy().tolist()))
        e2 = store.sample_layer(tg, fanout, seed, 2)
        e1 = store.sample_layer(np.unique(np.concatenate([tg, e2[:, 0]])), fanout, seed, 1)
        mine = set(map(tuple, np.concatenate([e1, e2]).tolist()))
        same = mine == ref and np.array_equal(np.unique(np.concatenate([np.unique(np.concatenate([tg, e2[:, 0]])), e1[:, 0]])), b1.src_nodes)
        print(f"[verify] batch {b}: rgcn {len(ref)} edges, vectorised {len(mine)} edges, identical {same}", flush=True)
        ok &= same
    return ok
