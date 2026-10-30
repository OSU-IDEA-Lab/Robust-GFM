""""Declared after training": compile every transformed variant back to one canonical program, on which an
unchanged pretrained model can be run.

Each variant's switchings are read from its relation names (the declaration):

  X__inverse / X__inv          named inverse of base relation X          -> edge removed (derivable)
  X__subject + X__object       X reified through a fact node             -> fact node removed, (h, X, t) restored
  a>>b                         2-hop shortcut (FB15k-237, Hetionet)      -> edge removed (derivable)
  __transitive_a|||b|||c       3-hop shortcut (WN18RR); a component may
                               end in __inv (traversed backwards)        -> edge removed (derivable)

Nothing is removed on trust: every removed edge is checked to be derivable from the base edges (its inverse exists,
or a path of the declared composition exists), and every fact node must have exactly one subject and one object
edge. Counts of verified edges go into the manifest; any unparsed or unverifiable edge stops the run.

The canonical program is the set of base-relation edges in lexicographic (h, r, t) order over the base entities and
relations. `original` is compiled the same way, so every `<variant>__compiled` should be byte-identical to
`original__compiled` (or `no_changes__compiled`). This is canonicalization at inference by declared switchings: it
works with an unmodified, relation-agnostic backbone and needs no retraining. (A richer invariant program, e.g. the
closure over all representations, would need a training run.)

    python compile_back.py --dataset FB15k237
"""

import argparse
import os
from collections import defaultdict

import numpy as np

from kg_format import file_sha256, load_base, load_variant, write_variant

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("GFM_DATA", os.path.join(HERE, "..", "data"))


def parse(name, base_rid):
    """-> ('base', r) | ('inverse', r) | ('subject', r) | ('object', r) | ('compose', [(r, forward), ...]) | None"""
    if name in base_rid:
        return ("base", base_rid[name])
    for suf in ("__inverse", "__inv"):
        if name.endswith(suf) and name[: -len(suf)] in base_rid:
            return ("inverse", base_rid[name[: -len(suf)]])
    for suf, kind in (("__subject", "subject"), ("__object", "object")):
        if name.endswith(suf) and name[: -len(suf)] in base_rid:
            return (kind, base_rid[name[: -len(suf)]])
    parts = None
    if ">>" in name:
        parts = name.split(">>")
    elif name.startswith("__transitive__") and "|||" in name:  # "__transitive_" + "_" + first component
        parts = name[len("__transitive__"):].split("|||")
    if parts:
        steps = []
        for p in parts:
            if p in base_rid:
                steps.append((base_rid[p], True))
            elif p.endswith("__inv") and p[: -len("__inv")] in base_rid:
                steps.append((base_rid[p[: -len("__inv")]], False))
            else:
                return None
        return ("compose", steps)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--variants", nargs="*", default=None, help="default: every non-compiled variant")
    args = ap.parse_args()
    ddir = os.path.join(args.data_root, args.dataset)
    base = load_base(ddir)
    N0, R0 = len(base["entities"]), len(base["relations"])
    base_rid = {r: i for i, r in enumerate(base["relations"])}
    names = args.variants or sorted(v for v in os.listdir(os.path.join(ddir, "variants")) if "__compiled" not in v)

    hashes = {}
    for vname in names:
        v = load_variant(ddir, vname)
        kinds = [parse(r, base_rid) for r in v["relations"]]
        g = v["graph"]
        stats = defaultdict(int)
        # 1. base edges among original entities; fact-node edges collected by node
        program, subj, obj, kept_unverified = set(), {}, {}, []
        derived = []  # (h, r, t, kind) to verify after the base program is known
        for h, r, t in g.tolist():
            k = kinds[r]
            if k is None:
                kept_unverified.append((h, r, t)); stats["unparsed_relation_edges_kept"] += 1
            elif k[0] == "base":
                program.add((h, k[1], t))
            elif k[0] == "subject":
                subj.setdefault(t, []).append((h, k[1]))
            elif k[0] == "object":
                obj.setdefault(h, []).append((t, k[1]))
            else:
                derived.append((h, k, t))
        # 2. un-reify: each fact node -> one base edge
        for n in set(subj) | set(obj):
            s, o = subj.get(n, []), obj.get(n, [])
            if len(s) == 1 and len(o) == 1 and s[0][1] == o[0][1] and n >= N0:
                program.add((s[0][0], s[0][1], o[0][0])); stats["fact_nodes_unreified"] += 1
            else:
                stats["fact_nodes_malformed"] += 1
        # 3. verify every derived edge against the base program, then drop it
        out_idx = defaultdict(set)  # (h, r) -> {t}
        in_idx = defaultdict(set)   # (t, r) -> {h}
        for h, r, t in program:
            out_idx[(h, r)].add(t)
            in_idx[(t, r)].add(h)

        def step(nodes, r, forward):
            res = set()
            for x in nodes:
                res |= out_idx.get((x, r), set()) if forward else in_idx.get((x, r), set())
            return res

        for h, k, t in derived:
            if k[0] == "inverse":
                ok = (t, k[1], h) in program
            else:
                steps = k[1]
                front = {h}
                for r, fw in steps[:-1]:
                    front = step(front, r, fw)
                    if not front:
                        break
                last_r, last_fw = steps[-1]
                preds = in_idx.get((t, last_r), set()) if last_fw else out_idx.get((t, last_r), set())
                ok = bool(front & preds)
            if ok:
                stats[f"{k[0]}_edges_verified_and_removed"] += 1
            else:
                stats[f"{k[0]}_edges_unverifiable_kept"] += 1
                kept_unverified.append((h, k, t))
        if kept_unverified or stats["fact_nodes_malformed"]:
            raise SystemExit(f"{vname}: {len(kept_unverified)} unparsed/unverifiable edges, "
                             f"{stats['fact_nodes_malformed']} malformed fact nodes; inspect before using")
        graph = np.array(sorted(program), dtype=np.int64).reshape(-1, 3)
        assert graph.size == 0 or (graph[:, [0, 2]].max() < N0 and graph[:, 1].max() < R0)
        m = write_variant(ddir, f"{vname}__compiled", base["entities"], base["relations"], graph, {
            "dataset": args.dataset, "transformation": f"compile_back({vname})",
            "compiled_from": vname, "compiled_from_hashes": v["manifest"]["hashes"],
            "declaration_read_from": "variant relation names", **{k: int(x) for k, x in stats.items()},
            "num_added_relations": 0, "num_added_entities": 0, "unanswerable_query_relations": []})
        hashes[vname] = m["hashes"]["graph.tsv"]
        print(f"{vname:>22} -> {m['num_graph_edges']} edges, graph {hashes[vname]}  {dict(stats)}")
    ref = "original" if "original" in hashes else "no_changes"
    same = {v: h == hashes.get(ref) for v, h in hashes.items()}
    print(f"byte-identical to {ref}__compiled:", same)


if __name__ == "__main__":
    main()
