#!/usr/bin/env python3
"""Compile SeHGNN channel bundles for every arm × flavour.

    --dataset IMDB     --variant {v1..v4,universal} --graph {physical,semantic}
                       reads RGCN_structural_v10_year_canonfeat/IMDB/nc/<variant> (physical graph, compiled
                       semantic program, canonical features, labels, split); K = 4, label channels, no cap
    --dataset FREEBASE --variant {unchanged,exact_2,exact_3,union_exact_2_3} --graph {physical,semantic}
                       physical = the variant's own relations; semantic = the family catalog (base ∪ registered
                       shortcuts of unchanged/exact_2/exact_3) realised per variant by MAGNN v2's
                       ``endpoint_pairs`` (direct edges if materialised, else the binarised composition);
                       K = 2, cap 500, RGCN split, labelled BOOK rows only
    --flavor full              IMDb: every type path ≤ K over the type-collapsed undirected adjacency (stock SeHGNN;
                               on IMDb one relation per typed pair, so this is relation-aware)
    --flavor full_rel          Freebase: relation-aware K-hop paths, every relation its own step
    (only these two channel sets are exposed; the other branches below are unreachable)

arm ↔ (graph, variant): originals = physical/<variant>; universal = physical/universal|union_exact_2_3;
invariant = semantic/<variant>. Output: <out-root>/<DATASET>/<flavor>/<graph>/<variant>/ (§5).
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
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "MAGNN")); sys.path.insert(0, str(_HERE.parent.parent / "common"))
from inv_sehgnn.channels import ChannelSpec, FLAVORS, compile_channel, full_type_paths, prefix_closure, save_bundle, typed_union  # noqa: E402
from magnn_compile_common import binarize, pair_hash  # noqa: E402

_DATA = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data")))
DEFAULT_OUT = _DATA / "preprocessed" / "sehgnn_v2"
IMDB_BUNDLES = _DATA / "preprocessed" / "RGCN_structural_v10_year_canonfeat" / "IMDB" / "nc"
FREEBASE_ROOT = _DATA / "dataset_variant_3hops_filtered"
IMDB_LETTER = {0: "M", 1: "D", 2: "A", 3: "L", 4: "Y"}
CONTRACT = {"IMDB": {"K": 4, "cap": 0, "label_hops": True}, "FREEBASE": {"K": 2, "cap": 500, "label_hops": False}}


def symmetric_steps(half: list[sp.csr_matrix]) -> list[sp.csr_matrix]:
    """MAGNN's symmetric metapath: steps to the middle, then their transposes back."""
    return list(half) + [binarize(m.T.tocsr()) for m in reversed(half)]


# ----------------------------------------------------------------------------- IMDb
def imdb(args):
    from inv_rgcn.data import load_graph_data
    from preprocess_imdb_magnn_v2 import TYPES, metapath_specs, relation_matrices

    d = load_graph_data(IMDB_BUNDLES / args.variant)
    n = d.num_nodes; node_type = d.node_type.numpy()
    if args.graph == "semantic":
        mats = relation_matrices(d.semantic_edge_index, d.semantic_edge_type, d.semantic_relation_names, n)
    else:
        mats = relation_matrices(d.physical_edge_index, d.physical_edge_type, d.physical_relation_names, n)
    rel_types = {name: (TYPES[name.split("_to_")[0]], TYPES[name.split("_to_")[1]]) for name in mats}
    targets = np.where(node_type == 0)[0]
    labels = d.y.numpy()[targets]
    split = {k: np.where(getattr(d, f"{s}_mask").numpy()[targets])[0] for k, s in (("train_idx", "train"), ("val_idx", "val"), ("test_idx", "test"))}
    K, cap = args.K, args.cap
    specs: list[ChannelSpec] = []
    label_paths: list[list[int]] = []
    if args.flavor in ("full", "full_rel"):
        if args.graph == "semantic" and args.skip_types:
            # skip nodes are CONTEXT only (docs/THEORY.md §1.6 / §2.4, MAGNN's M(L)D): the invariant channel set is the
            # stock type-path bias over the canonical schema realised on the semantic relations, and a skip
            # type never appears on a path -- neither as an interior node nor as a feature source. Without
            # this, every path through Link duplicates a path without it (Link is a bijection copy of Movie):
            # 143 channels carrying 21 distinct feature matrices .
            skip = {TYPES[t] for t in args.skip_types.split(",") if t}
            mats = {k: m for k, m in mats.items() if not (set(rel_types[k]) & skip)}
            rel_types = {k: v for k, v in rel_types.items() if k in mats}
        adj = typed_union(mats, rel_types, n)
        for path, steps in full_type_paths(adj, 0, K):
            key = "".join(IMDB_LETTER[t] for t in path)
            specs.append(ChannelSpec(key, path, steps, [f"{IMDB_LETTER[a]}-{IMDB_LETTER[b]}" for a, b in zip(path[:-1], path[1:])]))
            if path[-1] == 0:
                label_paths.append(path)
    else:
        set_mode = "union" if args.flavor == "restricted_union" else "intersection"
        stock = args.graph == "physical"
        mps = []
        for name, node_types, half in metapath_specs("movie", mats, set_mode, stock):
            rels = name.split("::")[1].split(";")
            step_names = rels + [r + "^T" for r in reversed(rels)]
            mps.append((name, node_types, symmetric_steps(half), step_names))
            label_paths.append(node_types)
        specs.append(ChannelSpec("M", [0], [], []))
        seen = {"M"}
        for name, j, node_types, steps, step_names in prefix_closure(mps):
            key = "".join(IMDB_LETTER[t] for t in node_types)
            if key in seen:
                continue                                    # e.g. 'ML' is the prefix of MLM and of MLDLM (same relation)
            seen.add(key)
            specs.append(ChannelSpec(key, node_types, steps, step_names))
    specs.sort(key=lambda s: (len(s.node_types), s.key))
    key_index = {s.key: i for i, s in enumerate(specs)}
    label_channels = []
    if CONTRACT["IMDB"]["label_hops"]:
        for path in sorted({tuple(p) for p in label_paths}, key=lambda p: (len(p), p)):
            key = "".join(IMDB_LETTER[t] for t in path)
            label_channels.append((key[:-1] + "Y", key_index[key]))
    features = {t: d.x.numpy()[node_type == t] for t in range(5)}
    graph_hash = hashlib.sha256("\n".join(f"{k}:{pair_hash(mats[k])}" for k in sorted(mats)).encode()).hexdigest()
    hashes = {"physical_hash": d.physical_hash, "semantic_hash": d.semantic_hash, "graph_hash": graph_hash,
              "bundle_feature_hash": d.feature_hash, "bundle_supervision_hash": d.supervision_hash, "relations": sorted(mats),
              "skip_types": args.skip_types if (args.graph == "semantic" and args.flavor == "full") else ""}
    return node_type, targets, labels, split, specs, label_channels, features, hashes, K, cap


# ----------------------------------------------------------------------------- Freebase
def freebase(args):
    from freebase_block_common import make_split
    from preprocess_freebase_magnn_v2 import CENTER_TYPE, Variant, book_view, build_catalog, endpoint_pairs, oriented_steps, relation_types

    cat = build_catalog(args.variants_root, tuple(args.family.split(",")))
    v = Variant(args.variants_root, args.variant)
    n = v.n; node_type = v.node_type
    y, target_nodes, local_split, _ = make_split(v.labels, node_type, CENTER_TYPE, 0.6, 0.2, args.split_seed)
    # relation set of the graph: physical = the variant's own; semantic = the whole family catalog realised on it
    if args.graph == "semantic":
        names = sorted(set(cat["base"]) | set(cat["shortcuts"]))
        mats, how = {}, {}
        for name in names:
            P, h = endpoint_pairs(v, name, cat); mats[name] = P; how[name] = h
        rel_types = {name: tuple(relation_types(name, cat)) for name in names}
    else:
        mats = dict(v.by_name); how = {k: "direct" for k in mats}
        rel_types = {rname: (st, en) for _, (rname, st, en) in v.rels.items()}
    print(f"[{v.name}] {args.graph}: {len(mats)} relations ({sum(h == 'composed' for h in how.values())} composed)", flush=True)
    K, cap = args.K, args.cap
    specs: list[ChannelSpec] = []
    if args.flavor == "full":
        adj = typed_union(mats, rel_types, n)
        for path, steps in full_type_paths(adj, CENTER_TYPE, K):
            key = "".join(str(t) for t in path)
            specs.append(ChannelSpec(key, path, steps, [f"{a}-{b}" for a, b in zip(path[:-1], path[1:])]))
    elif args.flavor == "full_rel":
        # relation-aware K-hop paths: every semantic relation is its own step (no type collapse). Stock
        # SeHGNN's Freebase loader merges all relations between two types into one channel; on the union
        # graph / semantic program that floods a base relation (MUSIC-in-BOOK, 31k pairs) with the composed
        # ones sharing its endpoint types (up to 234M pairs).
        steps_by_src: dict[int, list] = {}
        for name in sorted(mats):
            st, en = rel_types[name]
            steps_by_src.setdefault(st, []).append((f"+{name}", mats[name], en))
            steps_by_src.setdefault(en, []).append((f"-{name}", binarize(mats[name].T.tocsr()), st))
        specs.append(ChannelSpec(str(CENTER_TYPE), [CENTER_TYPE], [], []))
        frontier = [([CENTER_TYPE], [], [])]
        for _ in range(K):
            nxt = []
            for path, steps, names in frontier:
                for tok, M, dst in steps_by_src.get(path[-1], []):
                    p2, s2, n2 = path + [dst], steps + [M], names + [tok]
                    specs.append(ChannelSpec(">".join(n2) + f"#{dst}", p2, s2, n2)); nxt.append((p2, s2, n2))
            frontier = nxt
    else:
        incident = [nm for nm in sorted(mats) if CENTER_TYPE in rel_types[nm]]
        mps = []
        for rname in incident:                                            # skip form  B -r-> Y -r^-1-> B
            st, en = rel_types[rname]
            M, ytype = book_view(mats[rname], st, en, node_type)
            mps.append((rname, [CENTER_TYPE, int(ytype), CENTER_TYPE], symmetric_steps([M]), [rname, rname + "^T"]))
        if args.flavor == "restricted_union":
            if args.graph != "semantic":
                raise SystemExit("restricted_union (long forms over base relations) is defined for the semantic program only")
            for rname in incident:                                        # long form over base relations, interior real
                if rname not in cat["shortcuts"]:
                    continue
                half, types = oriented_steps(v, rname, cat)
                comp = cat["shortcuts"][rname]["composition"]
                names_half = list(comp) if cat["shortcuts"][rname]["start"] == CENTER_TYPE else [c + "^T" for c in reversed(comp)]
                node_types = [int(t) for t in types] + [int(t) for t in reversed(types[:-1])]
                mps.append(("long::" + rname, node_types, symmetric_steps(half), names_half + [s + "^T" for s in reversed(names_half)]))
        specs.append(ChannelSpec(str(CENTER_TYPE), [CENTER_TYPE], [], []))
        for name, j, node_types, steps, step_names in prefix_closure(mps):
            specs.append(ChannelSpec(f"{name}#{j}#{node_types[-1]}", node_types, steps, step_names))
    specs.sort(key=lambda s: (len(s.node_types), s.key))
    physical_hash = hashlib.sha256("\n".join(f"{k}:{pair_hash(m)}" for k, m in sorted(v.by_name.items())).encode()).hexdigest()
    semantic_hash = hashlib.sha256("\n".join(f"{k}:{pair_hash(mats[k])}" for k in sorted(mats)).encode()).hexdigest()
    hashes = {"physical_hash": physical_hash, "semantic_hash": semantic_hash if args.graph == "semantic" else None, "graph_hash": semantic_hash,
              "relations": sorted(mats), "realisation": how, "family": args.family, "split_seed": args.split_seed}
    return node_type, target_nodes, y, local_split, specs, [], None, hashes, K, cap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("IMDB", "FREEBASE"), required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--graph", choices=("physical", "semantic"), required=True)
    ap.add_argument("--flavor", choices=("full", "full_rel"), required=True)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--K", type=int, default=None); ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--cap-seed", type=int, default=1566911444)
    ap.add_argument("--variants-root", type=Path, default=FREEBASE_ROOT)
    ap.add_argument("--family", default="unchanged,exact_2,exact_3")
    ap.add_argument("--split-seed", type=int, default=1566911444)
    ap.add_argument("--skip-types", default="link", help="IMDb, semantic full flavour: node types treated as context-only (comma list; '' = none)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    c = CONTRACT[args.dataset]
    if args.K is None: args.K = c["K"]
    if args.cap is None: args.cap = c["cap"]
    out = args.out_root / args.dataset / args.flavor / args.graph / args.variant
    if (out / "manifest.json").exists() and not args.overwrite:
        print(f"[skip] {out} exists"); return
    t0 = time.time()
    node_type, targets, labels, split, specs, label_channels, features, hashes, K, cap = (imdb if args.dataset == "IMDB" else freebase)(args)
    print(f"[{args.dataset}/{args.variant}] {args.graph}/{args.flavor}: {len(specs)} channels, {len(label_channels)} label channels, targets {len(targets)} ({time.time()-t0:.0f}s)", flush=True)
    channels = []
    for i, spec in enumerate(specs):
        t1 = time.time()
        R, audit = compile_channel(spec, targets, node_type, cap, args.cap_seed)
        channels.append((spec, R, audit))
        print(f"  [{i:3d}] {spec.key:70s} types {spec.node_types} nnz {R.nnz:>9d} exact {audit['exact_rows']:>5d} sampled {audit['sampled_rows']:>5d} empty {audit['empty_rows']:>5d} ({time.time()-t1:.1f}s)", flush=True)
    m = save_bundle(out, dataset=args.dataset, variant=args.variant, graph=args.graph, flavor=args.flavor, K=K, cap=cap, cap_seed=args.cap_seed,
                    target_type=0, node_type=node_type, targets=targets, labels=labels, split=split, channels=channels,
                    label_channels=label_channels, features=features, hashes=hashes, t0=t0)
    print(f"[done] {out}: program {m['program_hash'][:12]} supervision {m['supervision_hash'][:12]} physical {str(m['physical_hash'])[:12]} "
          f"graph {m['graph_hash'][:12]} features {str(m['feature_hash'])[:12]} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
