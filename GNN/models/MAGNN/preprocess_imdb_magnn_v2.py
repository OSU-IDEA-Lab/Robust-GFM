#!/usr/bin/env python3
"""MAGNN IMDb preprocessing on the YEAR graph, from the RGCN structural bundles.

Tasks: ``nc`` (v1–v4 + universal), ``movie_director`` (v1, v3 + universal), ``movie_year``
(v1–v4 + universal). Reads ``RGCN_structural_v10_year_canonfeat/IMDB/<task>/<variant>``,
which already carries, per variant, the PHYSICAL graph, the compiled SEMANTIC program
(hash-identical across variants: the year node, the relocated director/actor and the
link-side projections are all in it), the canonical pooled-BoW features, the labels and
the shared splits / negatives. So the year node and the skip paths need no new hand-written
tables — a semantic relation's endpoint set IS its skip-compiled realisation.

Node types (bundle ids): 0 movie, 1 director, 2 actor, 3 link, 4 year.

Metapath sets (movie-centred for NC and for the LP head side; target-centred for the LP tail side):

* ``--arm stock`` (originals per variant / universal on the union graph), PHYSICAL edges:
  every symmetric 2-hop ``M–r→X–r⁻¹→M`` for a physical relation ``r`` out of movie, plus
  every link-mediated 4-hop ``M–L→X–L⁻¹→M`` for a physical ``link_to_X`` — the old
  ``preprocess_IMDB_star*.py`` convention (e.g. v2: MLDLM, MLALM, MLM) extended by year.
* ``--arm invariant --set-mode intersection``, SEMANTIC edges: the 2-hop forms only
  (MDM, MAM, MLM, MYM) — on every variant the semantic ``movie_to_director`` etc. are the
  compiled (skip) realisations, so the instance tables are identical by construction.
* ``--arm invariant --set-mode union``: 2-hop forms PLUS the link-mediated 4-hop forms over
  semantic edges (MLDLM, MLALM, MLYLM) — the old ``preprocess_IMDB_skip.py`` union design,
  with the link-side relations already collapsed by the compiler.

For LP tasks the held-out target relation (``movie_to_director`` / ``movie_to_year``) is
present in both graphs for TRAINING pairs only (``target_edges_hidden``), exactly as for
RGCN/CMPNN; the runner masks a training query's own instances (MAGNN's ``use_masks``).
The tail side gets the mirrored metapath set centred on the target type.

Instances: each exactly once (set semantics), canonical order, deterministic cap K per
(source, metapath) keyed on the semantic identity (``magnn_compile_common``). Output layout
= the Freebase v2 bundles (+ ``features_<t>.npy`` per node type, ``task_data.npz`` for LP),
consumed by ``run_magnn_nc.py`` / ``run_magnn_lp.py``.
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
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
from inv_rgcn.data import load_graph_data  # noqa: E402
from magnn_compile_common import binarize, pair_hash, sha, sample_instances, sample_long_paths  # noqa: E402

TYPES = {"movie": 0, "director": 1, "actor": 2, "link": 3, "year": 4}
TNAME = {v: k for k, v in TYPES.items()}
DEFAULT_ROOT = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data"))) / "preprocessed/RGCN_structural_v10_year_canonfeat/IMDB"
TARGET = {"movie_director": "movie_to_director", "movie_year": "movie_to_year", "nc": None}


def relation_matrices(edges_idx, edges_type, names, n):
    """name -> binarised CSR adjacency (src x dst) for every relation present."""
    out = {}
    src, dst, typ = edges_idx[0].numpy(), edges_idx[1].numpy(), edges_type.numpy()
    for r in np.unique(typ):
        m = typ == r
        out[names[int(r)]] = binarize(sp.coo_matrix((np.ones(int(m.sum()), dtype=np.int8), (src[m], dst[m])), shape=(n, n)))
    return out


def rel(names_present, a: str, b: str):
    return f"{a}_to_{b}" if f"{a}_to_{b}" in names_present else None


def metapath_specs(center: str, mats: dict, set_mode: str, stock: bool):
    """List of (name, node_types, [oriented step matrices out to the middle])."""
    specs = []
    present = set(mats)
    others = [t for t in TYPES if t != center]
    # 2-hop symmetric: C -r-> X -r^-1-> C
    for x in others:
        r = rel(present, center, x)
        if r is None and f"{x}_to_{center}" in present:      # only the reverse is named
            r = f"{x}_to_{center}"; M = binarize(mats[r].T.tocsr())
        elif r is not None:
            M = mats[r]
        else:
            continue
        specs.append((f"{center[0].upper()}{x[0].upper()}{center[0].upper()}::{r}", [TYPES[center], TYPES[x], TYPES[center]], [M]))
    # link-mediated 4-hop: C -link-> L -X-> X -X^-1-> L -link^-1-> C (center = movie only)
    if center == "movie" and (stock or set_mode == "union"):
        if "movie_to_link" in present:
            ML = mats["movie_to_link"]
            for x in ("director", "actor", "year"):
                r = rel(present, "link", x)
                if r is None:
                    continue
                specs.append((f"ML{x[0].upper()}LM::movie_to_link;{r}", [0, 3, TYPES[x], 3, 0], [ML, mats[r]]))
    return specs


def compile_center(center: str, mats: dict, sources: np.ndarray, set_mode: str, stock: bool, K: int, seed: int, out: Path, tag: str):
    specs = metapath_specs(center, mats, set_mode, stock)
    (out / "adjlist").mkdir(parents=True, exist_ok=True); (out / "idx").mkdir(exist_ok=True)
    defs, etype_base = [], 0
    for i, (name, node_types, steps) in enumerate(specs):
        if len(steps) == 1:
            M = steps[0]; MT = binarize(M.T.tocsr())
            inst, per = sample_instances(M, MT, sources, K, tag + name, seed)
            ph = pair_hash(M); npairs = int(M.nnz)
        else:
            inst, per = sample_long_paths(steps, sources, K, tag + name, seed)
            ph = hashlib.sha256("".join(pair_hash(m) for m in steps).encode()).hexdigest(); npairs = int(steps[-1].nnz)
        k_steps = len(node_types) - 1
        et = [etype_base + 2 * j for j in range(k_steps // 2)]; et = et + [e + 1 for e in reversed(et)]; etype_base += k_steps
        ih = sha(inst)
        np.save(out / "idx" / f"{i}.npy", inst.astype(np.int32))
        with (out / "adjlist" / f"{i}.txt").open("w") as f:
            k = 0
            for b, cnt in zip(sources, per):
                f.write(" ".join([str(int(b))] + [str(int(x)) for x in inst[k:k + cnt, -1]]) + "\n"); k += cnt
        defs.append({"index": i, "name": name, "node_types": node_types, "etypes": et, "pairs": npairs, "instances": int(inst.shape[0]),
                     "sources_with_instances": int((per > 0).sum()), "sources_capped": int((per >= K).sum()), "pair_hash": ph, "instance_hash": ih})
        print(f"    [{i}] {name:40s} width {len(node_types)} pairs {npairs:>7d} inst {inst.shape[0]:>8d} capped {(per >= K).sum():>5d}/{len(sources)} inst={ih[:10]}", flush=True)
    return defs, etype_base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--task", choices=("nc", "movie_director", "movie_year"), required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--arm", choices=("invariant", "stock"), required=True)
    ap.add_argument("--set-mode", choices=("intersection", "union"), default="intersection")
    ap.add_argument("--out", type=Path, required=True, help="<out>/<variant>/ is written")
    ap.add_argument("--cap", type=int, default=500); ap.add_argument("--cap-seed", type=int, default=1566911444)
    args = ap.parse_args()
    t0 = time.time()
    d = load_graph_data(args.bundle_root / args.task / args.variant)
    n = d.num_nodes; node_type = d.node_type.numpy()
    graph = "semantic" if args.arm == "invariant" else "physical"
    if graph == "semantic":
        mats = relation_matrices(d.semantic_edge_index, d.semantic_edge_type, d.semantic_relation_names, n)
    else:
        mats = relation_matrices(d.physical_edge_index, d.physical_edge_type, d.physical_relation_names, n)
    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{args.task}/{args.variant}] {args.arm}/{args.set_mode} graph={graph} relations present: {sorted(mats)}", flush=True)
    stock = args.arm == "stock"
    tag = f"{args.task}|{args.arm}|{args.set_mode}|"
    movies = np.where(node_type == 0)[0]
    groups = {}
    print("  movie-centred metapaths:", flush=True)
    defs0, ne0 = compile_center("movie", mats, movies, args.set_mode, stock, args.cap, args.cap_seed, out / "mode0", tag + "mode0|")
    groups["mode0"] = {"center": "movie", "num_metapaths": len(defs0), "num_edge_type": ne0, "etypes_list": [x["etypes"] for x in defs0],
                       "widths": [len(x["node_types"]) for x in defs0], "metapath_names": [x["name"] for x in defs0], "defs": defs0}
    target_rel = TARGET[args.task]
    if target_rel:
        ttype = "director" if args.task == "movie_director" else "year"
        tnodes = np.where(node_type == TYPES[ttype])[0]
        print(f"  {ttype}-centred metapaths (LP tail side):", flush=True)
        defs1, ne1 = compile_center(ttype, mats, tnodes, args.set_mode, stock, args.cap, args.cap_seed, out / "mode1", tag + "mode1|")
        groups["mode1"] = {"center": ttype, "num_metapaths": len(defs1), "num_edge_type": ne1, "etypes_list": [x["etypes"] for x in defs1],
                           "widths": [len(x["node_types"]) for x in defs1], "metapath_names": [x["name"] for x in defs1], "defs": defs1}
        td = {k: np.asarray(v) for k, v in d.task_data.items()}
        np.savez(out / "task_data.npz", **td)
    # features (canonical table), labels, split, node types
    x = d.x.numpy()
    for t in range(5):
        np.save(out / f"features_{t}.npy", x[node_type == t].astype(np.float32))
    np.save(out / "node_types.npy", node_type)
    if args.task == "nc":
        np.save(out / "labels.npy", d.y.numpy()[movies]); np.save(out / "target_nodes.npy", movies)
        np.savez(out / "split.npz", train_idx=np.where(d.train_mask.numpy()[movies])[0], val_idx=np.where(d.val_mask.numpy()[movies])[0],
                 test_idx=np.where(d.test_mask.numpy()[movies])[0])
    prog = hashlib.sha256("\n".join(f"{g}:{x['name']}:{x['pair_hash']}:{x['instance_hash']}" for g in groups for x in groups[g]["defs"]).encode()).hexdigest()
    summary = {"dataset": "IMDB", "task": args.task, "variant": args.variant, "arm": args.arm, "set_mode": args.set_mode, "graph": graph,
               "cap": args.cap, "cap_seed": args.cap_seed, "groups": {g: {k: v for k, v in groups[g].items() if k != "defs"} for g in groups},
               "program_hash": prog, "physical_hash": d.physical_hash, "semantic_hash": d.semantic_hash,
               "supervision_hash": d.supervision_hash, "feature_hash": d.feature_hash, "feature_table_hash": d.metadata.get("feature_table_hash"),
               "target_relation": target_rel, "seconds": time.time() - t0}
    (out / "metapaths.json").write_text(json.dumps(groups, indent=1)); (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{args.task}/{args.variant}] {args.arm}/{args.set_mode}: program {prog[:12]} physical {d.physical_hash[:12]} supervision {d.supervision_hash[:12]} features {str(d.feature_hash)[:12]} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
