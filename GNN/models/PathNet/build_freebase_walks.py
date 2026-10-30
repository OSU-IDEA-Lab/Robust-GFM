#!/usr/bin/env python3
"""PathNet walks on the Freebase stores (CPU; one job per graph, all seeds).

    python build_freebase_walks.py --graph physical --variant unchanged|exact_2|exact_3|union_exact_2_3
    python build_freebase_walks.py --graph semantic --variant unchanged|exact_2|exact_3

Walk graph = undirected 0/1 adjacency of every channel of the store (RGCN's Freebase stores — the same inputs
RGCN reads: physical per variant, `union_exact_2_3` for the universal arm, the compiled block-conditional program for the
invariant arm, hash-identical across the three variants). Walks: the MERW sampler of the PathNet port
(``pathnet/walks.py``: Perron vector by power iteration, per-node transition rows, ``RandomState(seed)``,
40 walks × length 4 from every labelled book, distance-to-target buckets) — the same code as the IMDb cell.
Output: ``$OUT/<graph>_<variant>/walks_seed<s>.npz`` (walks, dists, targets) + ``meta.json`` (pair count, hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent / "RGCN"), str(HERE.parent.parent / "common")):
    sys.path.insert(0, p)
from pathnet.walks import build_symmetric_adj, build_transition_rows, sample_walks  # noqa: E402  (PathNet port, read-only)
from freebase_samples import STORES  # noqa: E402
sys.path.insert(0, str(HERE)); from walk_dist import bfs_distance_buckets  # noqa: E402
from freebase_block_common import load_csr, load_json  # noqa: E402  (RGCN, read-only)

OUT = Path(os.environ.get("INV_GNN_DATA", str(Path(__file__).resolve().parent / "data"))) / "preprocessed/pathnet_inv/freebase_walks"
SEEDS = (1566911444, 20241017, 20251017)


def sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a)); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["physical", "semantic"], required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--num-walks", type=int, default=40); ap.add_argument("--walk-length", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--dist", choices=["earliest", "bfs"], default="earliest",
                    help="bfs = upstream's BFS distance to the target (output dir gets _bfs); earliest = earliest walk position (proxy)")
    ap.add_argument("--walk-type", choices=["merw", "rw"], default="merw", help="rw = upstream's degree-normalised walk (output dir gets _rw)")
    args = ap.parse_args()
    root = STORES[(args.graph, args.variant)]
    out = OUT / (f"{args.graph}_{args.variant}" + ("" if args.walk_type == "merw" else f"_{args.walk_type}") + ("_bfs" if args.dist == "bfs" else "")); out.mkdir(parents=True, exist_ok=True)
    cat = load_json(root / "relation_catalog.json"); hashes = load_json(root / "model_input_hashes.json")
    n = int(np.load(root / "node_types.npy", mmap_mode="r").size)
    targets = np.asarray(np.load(root / "target_nodes.npy"), dtype=np.int64)
    t0 = time.time()
    keys = []
    for rec in sorted(cat["channels"], key=lambda r: int(r["channel_id"])):
        indptr, indices = load_csr(root / rec["directory"], mmap=False)          # incoming CSR: row = dst, entries = src
        dst = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
        src = indices.astype(np.int64)
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        k = lo * n + hi
        keys.append(np.unique(k[lo != hi]))
        del indptr, indices, dst, src, lo, hi, k
    keys = np.unique(np.concatenate(keys))
    pairs = np.stack([keys // n, keys % n], 1); del keys
    pair_hash = sha(pairs)
    print(f"[walks] {args.graph}/{args.variant}: {len(pairs):,} undirected pairs ({time.time() - t0:.0f}s) hash {pair_hash[:12]}", flush=True)
    A = build_symmetric_adj(pairs, n); del pairs
    nbrs, probs = build_transition_rows(A, args.walk_type)
    print(f"[walks] transitions built ({time.time() - t0:.0f}s)", flush=True)
    meta = {"store": str(root), "store_combined_hash": hashes["combined_model_input_hash"], "undirected_pairs_hash": pair_hash,
            "num_nodes": n, "num_targets": int(len(targets)), "num_walks": args.num_walks, "walk_type": args.walk_type, "dist": args.dist, "walk_length": args.walk_length, "seeds": {}}
    for s in args.seeds:
        walks, dists = sample_walks(targets, nbrs, probs, args.num_walks, args.walk_length, np.random.RandomState(s))
        if args.dist == "bfs":
            dists = bfs_distance_buckets(walks, A)
        np.savez(out / f"walks_seed{s}.npz", walks=walks, dists=dists, targets=targets)
        meta["seeds"][str(s)] = sha(walks, dists)
        print(f"[walks] seed {s}: walk hash {meta['seeds'][str(s)][:12]} ({time.time() - t0:.0f}s)", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
