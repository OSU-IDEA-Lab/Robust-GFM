#!/usr/bin/env python3
"""MAGNN Freebase NC preprocessing, v2 — set-semantics skip metapaths with a deterministic cap.

Replaces the generated ``preprocess_scripts/freebase/.../preprocess_freebase_node.py``
enumerators, whose Python path enumeration (a) truncates at ``--max-paths-per-start`` in
physical adjacency order — binding for ~90 % of books on `unchanged` — and (b) keeps one
instance per interior witness after the skipped columns are dropped, so the same semantic
instance appears up to 535 times on `unchanged` and once on `exact_2`. Either alone makes
the per-node instance multisets differ across variants; MAGNN's intra-metapath softmax and
its frequency-weighted instance sampler then diverge, and Kendall τ < 1 follows.

Mechanism (docs/THEORY.md §2.2, skip-node metapaths, compiled with set semantics):

* A **semantic relation** ``r`` from BOOK is a base relation incident to BOOK or a
  registered shortcut (2- or 3-hop composition of base relations) incident to BOOK.
* On a variant, its **endpoint-pair set** ``P_r ⊆ BOOK × Y`` is the direct edge set when
  the variant materialises ``r``, else the *binarised* product of the base adjacency
  matrices along the registered composition (interior witnesses collapse — the skip).
  Both realisations are asserted equal when the variant has both (gauntlet).
* The **metapath** for ``r`` is the symmetric ``BOOK –r→ Y –r⁻¹→ BOOK``; its semantic
  instances are the triples ``(b, y, b')`` with ``P_r[b, y] = P_r[b', y] = 1``, each
  exactly once. Interior nodes of the composition never appear (context-only).
* Per (source book, metapath) at most ``K`` instances are kept, chosen **uniformly by a
  seeded RNG whose seed is a hash of (metapath name, book id)** over the instances in
  canonical (y, b') order — a deterministic function of the semantic instance set, so
  identical across variants, never of the physical layout.

Arms (same code path, different metapath sets):

* ``--arm invariant``: metapaths = every semantic relation in the FAMILY catalog
  (base ∪ all shortcuts registered for {unchanged, exact_2, exact_3}) incident to BOOK,
  realised per variant as above. ``--set-mode intersection`` (default) keeps one
  representation per relation (the skip form); ``--set-mode union`` ADDS, for every
  shortcut, its long form ``BOOK –r1→ X –r2→ Y –r2⁻¹→ X′ –r1⁻¹→ BOOK`` over BASE relations
  with the interior nodes real (5-/7-node instances) — the generated union scripts'
  design. Long forms are enumerated on every variant from the same base edges, so they
  are variant-independent by construction; instances are drawn by a hierarchical
  deterministic sampler (exact per-node path counts via sparse products, then a seeded
  uniform draw decoded hop by hop) — no path list is ever materialised.
* ``--arm stock``: metapaths = every relation the variant itself MATERIALISES that is
  incident to BOOK (originals / universal arms): stock MAGNN, no compositions.

Outputs (``<out>/<variant>/``): ``metapaths.json`` (catalog + per-metapath instance hashes
+ pair-set hashes), ``adjlist/<i>.txt`` and ``idx/<i>.npy`` (int32 ``[n, 3]`` global ids,
canonical order; row → adjlist entry), ``labels.npy``, ``target_nodes.npy`` (sorted labelled
BOOK ids), ``split.npz`` (RGCN's canonical partition: ``freebase_block_common.make_split``),
``node_types.npy``, ``summary.json`` with every hash the gauntlet compares.
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
sys.path.insert(0, str(_HERE.parent.parent / "common"))
from freebase_block_common import make_split  # noqa: E402  (the canonical Freebase split)

CENTER_TYPE = 0            # BOOK
DEFAULT_ROOT = Path(os.environ.get("INV_GNN_DATA", str(_HERE / "data"))) / "dataset_variant_3hops_filtered"
FAMILY = ("unchanged", "exact_2", "exact_3")


# ----------------------------------------------------------------------------- readers
def read_info(path: Path):
    """type id -> name; relation id -> (name, start_type, end_type)."""
    types, rels = {}, {}
    section = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("node.dat"):
            section = "node"; continue
        if s.startswith("link.dat"):
            section = "link"; continue
        if s.startswith("TYPE") or s.startswith("LINK") or s.startswith("---") or s.startswith("Attribute"):
            continue
        parts = [p for p in line.split("\t") if p != ""]
        if section == "node" and len(parts) >= 2 and parts[0].isdigit():
            types[int(parts[0])] = parts[-1].strip()
        elif section == "link" and len(parts) >= 4 and parts[0].isdigit():
            rels[int(parts[0])] = (parts[3].strip(), int(parts[1]), int(parts[2]))
    return types, rels


def read_nodes(path: Path) -> np.ndarray:
    ntype = {}
    for line in path.open():
        p = line.rstrip("\n").split("\t")
        ntype[int(p[0])] = int(p[-1])
    n = max(ntype) + 1
    out = np.full(n, -1, dtype=np.int64)
    for k, v in ntype.items():
        out[k] = v
    assert (out >= 0).all()
    return out


def read_links(path: Path):
    a = np.loadtxt(path, dtype=np.int64, usecols=(0, 1, 2))
    return a[:, 0], a[:, 1], a[:, 2]


def read_labels(path: Path):
    out = []
    for line in path.open():
        p = line.rstrip("\n").split("\t")
        out.append((int(p[0]), int(p[2]), int(p[3])))
    return out


def read_registry(path: Path):
    out = {}
    for line in path.open():
        if line.strip():
            d = json.loads(line)
            out[d["new_relation_name"]] = {
                "composition": [c.strip() for c in d["composition"].split(";")],
                "hop_count": int(d["hop_count"]), "start_type": d["start_type"], "end_type": d["end_type"],
            }
    return out


from magnn_compile_common import binarize, h64, pair_hash, sha, sample_instances, sample_long_paths  # noqa: E402


class Variant:
    def __init__(self, root: Path, name: str):
        self.name = name
        d = root / name
        self.types, self.rels = read_info(d / "info.dat")
        self.node_type = read_nodes(d / "node.dat")
        self.n = len(self.node_type)
        src, dst, rid = read_links(d / "link.dat")
        self.by_name = {}
        for r, (rname, st, en) in self.rels.items():
            m = rid == r
            adj = sp.coo_matrix((np.ones(int(m.sum()), dtype=np.int8), (src[m], dst[m])), shape=(self.n, self.n))
            self.by_name[rname] = binarize(adj)
        self.labels = read_labels(d / "label.dat")

    def has(self, rname: str) -> bool:
        return rname in self.by_name


def compose(v: Variant, composition: list[str]) -> sp.csr_matrix:
    """Binarised product of base adjacencies along a registered composition (skip interior)."""
    m = v.by_name[composition[0]]
    for step in composition[1:]:
        m = binarize(m @ v.by_name[step])
    return m


def endpoint_pairs(v: Variant, rname: str, catalog: dict) -> tuple[sp.csr_matrix, str]:
    """P_r on variant v: direct edges if materialised, else the composition. Returns (matrix, how)."""
    if v.has(rname):
        direct = v.by_name[rname]
        if rname in catalog["shortcuts"]:      # gauntlet: materialised shortcut == composition
            comp = compose(v, catalog["shortcuts"][rname]["composition"])
            if pair_hash(direct) != pair_hash(comp):
                raise RuntimeError(f"{v.name}: materialised {rname} ({direct.nnz}) != composition ({comp.nnz})")
        return direct, "direct"
    if rname in catalog["shortcuts"]:
        return compose(v, catalog["shortcuts"][rname]["composition"]), "composed"
    raise KeyError(f"{v.name} cannot realise {rname}")


def book_view(P: sp.csr_matrix, start_type: int, end_type: int, node_type: np.ndarray):
    """Orient P so rows are BOOK: (BOOK x Y matrix restricted to BOOK rows, Y type)."""
    if start_type == CENTER_TYPE:
        M, ytype = P, end_type
    elif end_type == CENTER_TYPE:
        M, ytype = P.T.tocsr(), start_type
    else:
        raise ValueError("relation not incident to BOOK")
    return binarize(M), ytype


# ----------------------------------------------------------------------------- instances
def oriented_steps(v: Variant, rname: str, cat: dict):
    """Base-relation step matrices along the composition, oriented so the walk starts at BOOK."""
    d = cat["shortcuts"][rname]
    steps = []
    for name in d["composition"]:
        st, en = cat["base"][name]
        steps.append((v.by_name[name], st, en))
    if d["start"] == CENTER_TYPE:
        mats = [binarize(m) for m, _, _ in steps]; types = [st for _, st, _ in steps] + [steps[-1][2]]
    else:                                   # relation ends at BOOK: walk it backwards
        mats = [binarize(m.T.tocsr()) for m, _, _ in reversed(steps)]
        types = [en for _, _, en in reversed(steps)] + [steps[0][1]]
    return mats, types                      # mats[j]: types[j] -> types[j+1]



# ----------------------------------------------------------------------------- main
def build_catalog(root: Path, family: tuple[str, ...]):
    base_types, base_rels = read_info(root / "unchanged" / "info.dat")
    cat = {"types": base_types, "base": {name: (st, en) for _, (name, st, en) in base_rels.items()}, "shortcuts": {}}
    for v in family:
        reg = root / v / "new_transitive_relations.jsonl"
        if reg.exists():
            for name, d in read_registry(reg).items():
                cat["shortcuts"].setdefault(name, d)
    tname = {n: t for t, n in base_types.items()}
    for name, d in cat["shortcuts"].items():
        d["start"], d["end"] = tname[d["start_type"]], tname[d["end_type"]]
    return cat


def relation_types(name: str, cat: dict):
    if name in cat["base"]:
        return cat["base"][name]
    d = cat["shortcuts"][name]
    return d["start"], d["end"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--arm", choices=("invariant", "stock"), required=True)
    ap.add_argument("--set-mode", choices=("intersection", "union"), default="intersection")
    ap.add_argument("--family", default=",".join(FAMILY), help="variants whose registries define the semantic catalog")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=500, help="max instances per (book, metapath)")
    ap.add_argument("--cap-seed", type=int, default=1566911444)
    ap.add_argument("--split-seed", type=int, default=1566911444)
    ap.add_argument("--sources", choices=("labeled", "all"), default="labeled",
                    help="which BOOK nodes get instance rows (MAGNN only ever queries labelled targets)")
    args = ap.parse_args()
    t0 = time.time()
    cat = build_catalog(args.variants_root, tuple(args.family.split(",")))
    v = Variant(args.variants_root, args.variant)
    print(f"[{v.name}] nodes {v.n}, relations {len(v.rels)}, catalog base {len(cat['base'])} shortcuts {len(cat['shortcuts'])} ({time.time()-t0:.0f}s)", flush=True)

    # split (canonical RGCN partition) and sources
    y, target_nodes, local_split, global_split = make_split(v.labels, v.node_type, CENTER_TYPE, 0.6, 0.2, args.split_seed)
    sources = target_nodes if args.sources == "labeled" else np.where(v.node_type == CENTER_TYPE)[0]

    # metapath set
    if args.arm == "invariant":
        names = [n for n in list(cat["base"]) + list(cat["shortcuts"]) if CENTER_TYPE in relation_types(n, cat)]
    else:
        names = [rname for _, (rname, st, en) in sorted(v.rels.items()) if CENTER_TYPE in (st, en)]
    names = sorted(set(names))
    # metapath specs: ("skip", relation) for the symmetric skip form; ("long", relation) for the
    # union mode's long form of a shortcut relation (over base relations, interior nodes real)
    specs = [("skip", n) for n in names]
    if args.arm == "invariant" and args.set_mode == "union":
        specs += [("long", n) for n in names if n in cat["shortcuts"]]
    out = args.out / v.name
    (out / "adjlist").mkdir(parents=True, exist_ok=True); (out / "idx").mkdir(exist_ok=True)
    defs, inst_hashes, pair_hashes = [], [], []
    etype_base = 0
    for i, (form, rname) in enumerate(specs):
        st, en = relation_types(rname, cat)
        if form == "skip":
            P, how = endpoint_pairs(v, rname, cat)
            M, ytype = book_view(P, st, en, v.node_type)
            MT = binarize(M.T.tocsr())
            inst, per_src = sample_instances(M, MT, sources, args.cap, rname, args.cap_seed)
            ph = pair_hash(M); node_types = [CENTER_TYPE, int(ytype), CENTER_TYPE]; npairs = int(M.nnz)
        else:
            mats, types = oriented_steps(v, rname, cat)
            inst, per_src = sample_long_paths(mats, sources, args.cap, "long::" + rname, args.cap_seed)
            how = "long"; ph = hashlib.sha256("".join(pair_hash(m) for m in mats).encode()).hexdigest()
            node_types = [int(t) for t in types] + [int(t) for t in reversed(types[:-1])]; npairs = int(mats[-1].nnz)
        k_steps = len(node_types) - 1
        etypes = [etype_base + 2 * j for j in range(k_steps // 2)]          # forward steps: even ids
        etypes = etypes + [e + 1 for e in reversed(etypes)]                  # reverse steps: their conjugates
        etype_base += k_steps                                               # 2 ids per step pair
        ih = sha(inst)
        np.save(out / "idx" / f"{i}.npy", inst.astype(np.int32))
        with (out / "adjlist" / f"{i}.txt").open("w") as f:
            k = 0
            for b, cnt in zip(sources, per_src):
                f.write(" ".join([str(int(b))] + [str(int(x)) for x in inst[k:k + cnt, -1]]) + "\n"); k += cnt
        defs.append({"index": i, "name": rname if form == "skip" else "long::" + rname, "form": form, "how": how,
                     "node_types": node_types, "etypes": etypes,
                     "composition": cat["shortcuts"].get(rname, {}).get("composition"), "pairs": npairs,
                     "instances": int(inst.shape[0]), "sources_with_instances": int((per_src > 0).sum()),
                     "sources_capped": int((per_src >= args.cap).sum()), "pair_hash": ph, "instance_hash": ih})
        inst_hashes.append(ih); pair_hashes.append(ph)
        print(f"  [{i:2d}] {defs[-1]['name']:54s} {how:8s} width {len(node_types)} pairs {npairs:>9d} inst {inst.shape[0]:>9d} capped {(per_src >= args.cap).sum():>5d}/{len(sources)} inst={ih[:10]}", flush=True)
    names = [d["name"] for d in defs]

    np.save(out / "labels.npy", y); np.save(out / "target_nodes.npy", target_nodes); np.save(out / "node_types.npy", v.node_type)
    np.savez(out / "split.npz", **local_split, **{k + "_global": val for k, val in global_split.items()})
    summary = {
        "variant": v.name, "arm": args.arm, "set_mode": args.set_mode, "family": args.family, "cap": args.cap, "cap_seed": args.cap_seed,
        "split_seed": args.split_seed, "sources": args.sources, "num_sources": int(len(sources)),
        "num_metapaths": len(names), "metapath_names": names, "num_edge_type": etype_base,
        "etypes_list": [d["etypes"] for d in defs], "widths": [len(d["node_types"]) for d in defs],
        "program_hash": hashlib.sha256("\n".join(f"{n}:{p}:{h}" for n, p, h in zip(names, pair_hashes, inst_hashes)).encode()).hexdigest(),
        "pairs_hash": hashlib.sha256("\n".join(pair_hashes).encode()).hexdigest(),
        "physical_hash": hashlib.sha256("\n".join(f"{n}:{pair_hash(m)}" for n, m in sorted(v.by_name.items())).encode()).hexdigest(),
        "supervision_hash": sha(np.concatenate([target_nodes, y] + [global_split[k] for k in ("train_idx", "val_idx", "test_idx")])),
        "target_set_hash": sha(target_nodes)[:12], "seconds": time.time() - t0,
    }
    (out / "metapaths.json").write_text(json.dumps({"catalog": {"base": cat["base"], "shortcuts": cat["shortcuts"]}, "metapaths": defs}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{v.name}] {args.arm}: {len(names)} metapaths, program_hash {summary['program_hash'][:12]} physical {summary['physical_hash'][:12]} "
          f"supervision {summary['supervision_hash'][:12]} target_set {summary['target_set_hash']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
