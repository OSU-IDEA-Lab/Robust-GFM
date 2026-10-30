#!/usr/bin/env python3
"""MAGNN DBLP Paper–Conference LP preprocessing from the RGCN structural bundles.

Reads ``RGCN_structural_v8_dblp/DBLP/paper_conference/<variant>`` (v1 / v2 / v3 / universal),
the bundles RGCN and CMPNN already consume: per variant the PHYSICAL graph (Area on Paper /
Conference / Author; universal = all three), the compiled SEMANTIC program (hash-identical
across variants: ``paper_to_area`` / ``conference_to_area`` / ``author_to_area`` reconstructed
through the variant's own context, interior witnesses collapsed), the canonical split and its
K=19 negatives (= every other conference: the full 20-conference ranking protocol). Area edges
exist for TRAINING-support papers only in every variant (a held-out paper's area would leak
its conference), and ``paper_to_conference`` carries training pairs only.

Node types (bundle ids): 0 author, 1 paper, 2 term, 3 conference, 4 area.

Metapath catalog (DBLP Paper–Venue):

  paper-centred (mode0):  PAP, PTP, PVP, PArP, PAArAP, PVArVP
  conference-centred (mode1, the LP tail side):  VPV, VPAPV

* ``--arm stock`` (originals per variant / universal on the union graph), PHYSICAL edges: every
  catalog metapath whose relations are all physically present — v1: PArP; v2: PVArVP;
  v3: PAArAP; universal: all six. No skip nodes: the universal graph materialises every
  relation directly, so its stock set is the union of the per-variant original sets.
* ``--arm invariant --set-mode intersection``, SEMANTIC edges: the 2-hop forms PAP, PTP, PVP,
  PArP — ``PArP`` over the semantic ``paper_to_area`` is the skip realisation
  (v2: P(V)Ar(V)P, v3: P(A)Ar(A)P), identical across variants by construction.
* ``--arm invariant --set-mode union``: the 2-hop forms PLUS PAArAP and PVArVP over the semantic
  ``author_to_area`` / ``conference_to_area`` (the PA(P)Ar(P)AP / PV(P)Ar(P)VP forms).

The tail side (VPV, VPAPV) uses relations that no switching touches, so it is the same in
every arm. Instances: each exactly once (set semantics), canonical order, deterministic cap K
per (source, metapath) keyed on the semantic identity (``magnn_compile_common``). Features:
none stored — the runner uses sparse identity features per type (feature type 0, the
MAGNN DBLP contract). Output layout = the IMDb v2 LP bundles (``mode0/``, ``mode1/``,
``task_data.npz`` with per-positive negative tails, ``node_types.npy``, ``summary.json``),
consumed by ``run_magnn_lp.py --dataset DBLP`` and ``run_magnn_lp_augmentation.py``.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent.parent / "common"))
from inv_rgcn.data import load_graph_data  # noqa: E402
from magnn_compile_common import pair_hash, sha, sample_instances, sample_long_paths, binarize  # noqa: E402
from preprocess_imdb_magnn_v2 import relation_matrices  # noqa: E402

TYPES = {"author": 0, "paper": 1, "term": 2, "conference": 3, "area": 4}
DEFAULT_ROOT = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data"))) / "preprocessed/RGCN_structural_v8_dblp/DBLP"
TASK = "paper_conference"
TARGET_REL = "paper_to_conference"

# name -> (node types, oriented relations out to the middle). Symmetric metapaths: the second
# half is the transposes of the first half (magnn_compile_common).
CATALOG = {
    "mode0": [
        ("PAP", [1, 0, 1], ["paper_to_author"]),
        ("PTP", [1, 2, 1], ["paper_to_term"]),
        ("PVP", [1, 3, 1], ["paper_to_conference"]),
        ("PArP", [1, 4, 1], ["paper_to_area"]),
        ("PAArAP", [1, 0, 4, 0, 1], ["paper_to_author", "author_to_area"]),
        ("PVArVP", [1, 3, 4, 3, 1], ["paper_to_conference", "conference_to_area"]),
    ],
    "mode1": [
        ("VPV", [3, 1, 3], ["conference_to_paper"]),
        ("VPAPV", [3, 1, 0, 1, 3], ["conference_to_paper", "paper_to_author"]),
    ],
}
LONG_FORMS = {"PAArAP", "PVArVP"}


def metapath_specs(group: str, mats: dict, set_mode: str, stock: bool):
    """(name, node_types, [step matrices]) for the catalog entries this arm realises."""
    specs = []
    for name, node_types, rels in CATALOG[group]:
        if not stock and set_mode == "intersection" and name in LONG_FORMS:
            continue
        if any(r not in mats for r in rels):
            if stock:            # a relation the variant does not materialise: no such metapath
                continue
            raise SystemExit(f"semantic program lacks {rels} needed by {name}")
        specs.append((f"{name}::{';'.join(rels)}", node_types, [mats[r] for r in rels]))
    return specs


def compile_group(group: str, mats: dict, sources: np.ndarray, set_mode: str, stock: bool, K: int, seed: int, out: Path, tag: str):
    specs = metapath_specs(group, mats, set_mode, stock)
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
        print(f"    [{i}] {name:52s} width {len(node_types)} pairs {npairs:>7d} inst {inst.shape[0]:>8d} capped {(per >= K).sum():>5d}/{len(sources)} inst={ih[:10]}", flush=True)
    return defs, etype_base


def group_negatives(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Flat [N*K, 2] negatives (grouped by head paper) -> [N, K] tails aligned with ``pos`` rows."""
    if neg.ndim == 2 and neg.shape[1] > 2:
        return neg
    per = {}
    for h, t in neg:
        per.setdefault(int(h), []).append(int(t))
    K = {len(v) for v in per.values()}
    if len(K) != 1:
        raise SystemExit(f"negatives per paper not constant: {sorted(K)}")
    K = K.pop()
    out = np.empty((len(pos), K), dtype=np.int64)
    for i, (h, t) in enumerate(pos):
        tails = np.asarray(sorted(per[int(h)]), dtype=np.int64)
        if int(t) in set(tails.tolist()):
            raise SystemExit(f"positive tail {t} among the negatives of paper {h}")
        out[i] = tails
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--variant", required=True, choices=("v1", "v2", "v3", "universal"))
    ap.add_argument("--arm", choices=("invariant", "stock"), required=True)
    ap.add_argument("--set-mode", choices=("intersection", "union"), default="intersection")
    ap.add_argument("--out", type=Path, required=True, help="<out>/<variant>/ is written")
    ap.add_argument("--cap", type=int, default=500); ap.add_argument("--cap-seed", type=int, default=1566911444)
    args = ap.parse_args()
    t0 = time.time()
    d = load_graph_data(args.bundle_root / TASK / args.variant)
    n = d.num_nodes; node_type = d.node_type.numpy()
    graph = "semantic" if args.arm == "invariant" else "physical"
    if graph == "semantic":
        mats = relation_matrices(d.semantic_edge_index, d.semantic_edge_type, d.semantic_relation_names, n)
    else:
        mats = relation_matrices(d.physical_edge_index, d.physical_edge_type, d.physical_relation_names, n)
    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{TASK}/{args.variant}] {args.arm}/{args.set_mode} graph={graph} relations present: {sorted(mats)}", flush=True)
    stock = args.arm == "stock"
    # cap RNG keyed on the semantic identity only (dataset, task, group, metapath, source) -- never on
    # the arm: the same walk set draws the same capped subset in every arm (THEORY.md section 5)
    tag = f"DBLP|{TASK}|"
    groups = {}
    for g, center in (("mode0", "paper"), ("mode1", "conference")):
        srcs = np.where(node_type == TYPES[center])[0]
        print(f"  {center}-centred metapaths ({g}):", flush=True)
        defs, ne = compile_group(g, mats, srcs, args.set_mode, stock, args.cap, args.cap_seed, out / g, tag + g + "|")
        groups[g] = {"center": center, "num_metapaths": len(defs), "num_edge_type": ne, "etypes_list": [x["etypes"] for x in defs],
                     "widths": [len(x["node_types"]) for x in defs], "metapath_names": [x["name"] for x in defs], "defs": defs}
    td = {k: np.asarray(v) for k, v in d.task_data.items()}
    task = {}
    for split in ("train", "val", "test"):
        pos = td[f"{split}_pos"].astype(np.int64)
        task[f"{split}_pos"] = pos; task[f"{split}_neg"] = group_negatives(pos, td[f"{split}_neg"])
        assert (node_type[pos[:, 0]] == TYPES["paper"]).all() and (node_type[pos[:, 1]] == TYPES["conference"]).all()
    np.savez(out / "task_data.npz", **task)
    np.save(out / "node_types.npy", node_type)
    prog = hashlib.sha256("\n".join(f"{g}:{x['name']}:{x['pair_hash']}:{x['instance_hash']}" for g in groups for x in groups[g]["defs"]).encode()).hexdigest()
    sup = hashlib.sha256(b"".join(np.ascontiguousarray(task[k]).tobytes() for k in sorted(task))).hexdigest()
    summary = {"dataset": "DBLP", "task": TASK, "variant": args.variant, "arm": args.arm, "set_mode": args.set_mode, "graph": graph,
               "cap": args.cap, "cap_seed": args.cap_seed, "groups": {g: {k: v for k, v in groups[g].items() if k != "defs"} for g in groups},
               "program_hash": prog, "physical_hash": d.physical_hash, "semantic_hash": d.semantic_hash,
               "supervision_hash": d.supervision_hash, "task_data_hash": sup, "feature_hash": "identity", "feature_table_hash": None,
               "target_relation": TARGET_REL, "negatives_per_positive": int(task["test_neg"].shape[1]),
               "bundle": str(args.bundle_root / TASK / args.variant), "bundle_metadata": d.metadata, "seconds": time.time() - t0}
    (out / "metapaths.json").write_text(json.dumps(groups, indent=1)); (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"[{TASK}/{args.variant}] {args.arm}/{args.set_mode}: program {prog[:12]} physical {d.physical_hash[:12]} supervision {d.supervision_hash[:12]} "
          f"task_data {sup[:12]} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
