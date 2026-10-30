"""Build representation variants of a raw-triple KG (format: kg_format.py).

Used for FB15k-237 and Hetionet (WN18RR reuses the GNN variants, build_wn18rr_variants.py). Every variant
is built from the official TRAINING triples only; validation and test triples are queries and never enter a
message graph. Base entity and relation ids are fixed across variants. Each variant is an information-
preserving transformation (a switching):

  original            the official training graph
  inverse             + a NAMED inverse edge (t, r__inverse, h) per training edge. For a model that adds unnamed
                      inverses internally (e.g. ULTRA) this restates information it computes anyway
  reified_<set>       every edge (h, r, t) of the relations in <set> is replaced by a fact node n and two edges
                      (h, r__subject, n), (n, r__object, t) -- the IMDb Link-node / Freebase mediator pattern.
                      The direct r edges are gone, so queries on r cannot be posed natively on this variant;
                      they are listed as unanswerable and excluded from the common query subset. Sets:
                        cvt_strong  FB15k-237 only: the K most frequent relations whose names contain '.', i.e.
                                    the Freebase mediator (CVT) paths the dataset's creators collapsed into one
                                    relation; reifying them restores Freebase's own encoding
                        cvt_light   FB15k-237 only: the K mediator relations with the most training edges per
                                    test query (transforms a lot of context, removes almost no queries)
                        <name>      any set given with --reify-set name=rel1,rel2,...
  shortcut2_c<cap>    + materialized 2-hop shortcut edges (h, r1>>r2, t) for the top-K compositions by path
                      count (hub-dominated compositions above --shortcut-max-paths skipped), at most <cap>
                      edges per composition chosen by a keyed (identity-based) rule; several caps = a
                      dose-response series. <cap> accepts k / m suffixes (5k, 50k, 200k)

Hetionet-size test sets can be sampled (--max-queries): the sample is keyed on the query identity, and the
left-out held-out triples are kept for filtering only (base/filter_extra.tsv).
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np

from kg_format import key_seed, read_lines, write_base, write_variant

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("GFM_DATA", os.path.join(HERE, "..", "data"))


def read_raw(path):
    return [tuple(line.split()) for line in read_lines(path) if line.strip()]


def build_base(raw_dir):
    train, valid, test = (read_raw(os.path.join(raw_dir, f"{s}.txt")) for s in ("train", "valid", "test"))
    entities, relations, eid, rid = [], [], {}, {}
    for split in (train, valid, test):  # first-appearance order: train vocabulary first
        for h, r, t in split:
            for e in (h, t):
                if e not in eid:
                    eid[e] = len(entities)
                    entities.append(e)
            if r not in rid:
                rid[r] = len(relations)
                relations.append(r)

    def ids(split):
        return np.array([(eid[h], rid[r], eid[t]) for h, r, t in split], dtype=np.int64)

    return entities, relations, ids(train), ids(valid), ids(test)


def sample_queries(q, n, seed, split):
    if not n or len(q) <= n:
        return q, np.zeros((0, 3), dtype=np.int64)
    keys = np.array([key_seed(seed, "sample", split, h, r, t) for h, r, t in q.tolist()], dtype=np.uint64)
    order = np.argsort(keys, kind="stable")
    keep = np.sort(order[:n])
    mask = np.zeros(len(q), dtype=bool)
    mask[keep] = True
    return q[mask], q[~mask]


def parse_cap(s):
    s = s.lower()
    mult = 1_000_000 if s.endswith("m") else 1_000 if s.endswith("k") else 1
    return int(float(s.rstrip("km")) * mult)


def variant_inverse(entities, relations, train):
    R = len(relations)
    inv = train[:, [2, 1, 0]].copy()
    inv[:, 1] += R
    return entities, relations + [f"{r}__inverse" for r in relations], np.concatenate([train, inv]), {
        "num_added_relations": R, "num_added_entities": 0, "num_added_edges": int(len(inv)),
        "unanswerable_query_relations": []}


def choose_reify_set(name, relations, train, test, k, custom):
    ctr, cte = Counter(train[:, 1].tolist()), Counter(test[:, 1].tolist())
    if name == "cvt_strong":
        dotted = [i for i, r in enumerate(relations) if "." in r]
        return sorted(dotted, key=lambda i: (-ctr[i], i))[:k], "K most frequent Freebase mediator (dotted) relations"
    if name == "cvt_light":
        dotted = [i for i, r in enumerate(relations) if "." in r]
        return (sorted(dotted, key=lambda i: (-(ctr[i] / (cte[i] + 1)), i))[:k],
                "K mediator relations with the most training edges per test query")
    if name in custom:
        rid = {r: i for i, r in enumerate(relations)}
        missing = [r for r in custom[name] if r not in rid]
        assert not missing, f"--reify-set {name}: unknown relations {missing}"
        return [rid[r] for r in custom[name]], "user-specified set"
    raise ValueError(f"unknown reification set {name!r}")


def variant_reified(entities, relations, train, test, chosen, rule):
    chosen_set = set(chosen)
    ctr, cte = Counter(train[:, 1].tolist()), Counter(test[:, 1].tolist())
    rels, ents = list(relations), list(entities)
    new_rel = {}
    for r in chosen:
        new_rel[r] = (len(rels), len(rels) + 1)
        rels += [f"{relations[r]}__subject", f"{relations[r]}__object"]
    keep, added = [], []
    for h, r, t in train.tolist():
        if r in chosen_set:
            n = len(ents)
            ents.append(f"__fact__{relations[r]}__{entities[h]}__{entities[t]}")
            s, o = new_rel[r]
            added += [(h, s, n), (n, o, t)]
        else:
            keep.append((h, r, t))
    lost = sum(cte[r] for r in chosen)
    return ents, rels, np.array(keep + added, dtype=np.int64), {
        "selection_rule": rule,
        "reified_relations": [relations[r] for r in chosen],
        "reified_relation_ids": chosen,
        "reified_train_edges": {relations[r]: ctr[r] for r in chosen},
        "reified_test_queries": {relations[r]: cte[r] for r in chosen},
        "share_of_test_queries_unanswerable": lost / max(1, len(test)),
        "num_added_relations": 2 * len(chosen), "num_added_entities": len(ents) - len(entities),
        "num_removed_edges": int(len(train) - len(keep)), "num_added_edges": len(added),
        "unanswerable_query_relations": chosen,
    }


class ShortcutIndex:
    """Edges grouped by middle node and relation; path counts per 2-hop composition (computed once)."""

    def __init__(self, train):
        self.out = defaultdict(lambda: defaultdict(list))  # m -> r2 -> [t]
        self.inn = defaultdict(lambda: defaultdict(list))  # m -> r1 -> [h]
        for h, r, t in train.tolist():
            self.out[h][r].append(t)
            self.inn[t][r].append(h)
        self.path_counts = Counter()
        for m, ins in self.inn.items():
            outs = self.out.get(m)
            if not outs:
                continue
            for r1, hs in ins.items():
                for r2, ts in outs.items():
                    self.path_counts[(r1, r2)] += len(hs) * len(ts)
        self._pairs = {}

    def choose(self, k, max_paths):
        eligible = [(c, n) for c, n in self.path_counts.items() if n <= max_paths]
        return [c for c, _ in sorted(eligible, key=lambda x: (-x[1], x[0]))[:k]]

    def pairs(self, r1, r2):
        if (r1, r2) not in self._pairs:
            s = set()
            for m, ins in self.inn.items():
                hs = ins.get(r1)
                ts = self.out[m].get(r2) if m in self.out else None
                if hs and ts:
                    s.update((h, t) for h in hs for t in ts if h != t)
            self._pairs[(r1, r2)] = sorted(s)
        return self._pairs[(r1, r2)]


def variant_shortcut2(entities, relations, train, heldout, index, k, cap, max_paths, seed):
    chosen = index.choose(k, max_paths)
    rels, added, per_comp = list(relations), [], {}
    for (r1, r2) in chosen:
        cid = len(rels)
        rels.append(f"{relations[r1]}>>{relations[r2]}")
        ps = index.pairs(r1, r2)
        if len(ps) > cap:  # keyed: an edge's fate depends on its identity, never on its position
            ps = sorted(sorted(ps, key=lambda p: key_seed(seed, r1, r2, p[0], p[1]))[:cap])
        added += [(h, cid, t) for h, t in ps]
        per_comp[rels[-1]] = {"paths": int(index.path_counts[(r1, r2)]), "unique_pairs": len(index.pairs(r1, r2)),
                              "kept_edges": len(ps)}
    held = {(h, t) for h, _, t in heldout.tolist()}
    held |= {(t, h) for h, t in list(held)}
    sc_pairs = {(h, t) for h, _, t in added}
    return entities, rels, np.concatenate([train, np.array(added, dtype=np.int64).reshape(-1, 3)]), {
        "compositions": per_comp, "num_added_relations": len(chosen), "num_added_entities": 0,
        "num_added_edges": len(added), "cap_per_composition": cap, "max_paths_per_eligible_composition": max_paths,
        "num_compositions_skipped_as_hub_dominated": sum(1 for n in index.path_counts.values() if n > max_paths),
        "selection_seed": seed, "derived_from": "official train triples only",
        "shortcut_pairs_matching_a_heldout_pair_either_direction": len(sc_pairs & held),
        "unanswerable_query_relations": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="name; <data-root>/<dataset>/raw/{train,valid,test}.txt")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--reify-set", action="append", default=[], help="name=rel1,rel2,... (reified_<name>)")
    ap.add_argument("--reify-top-k", type=int, default=10)
    ap.add_argument("--shortcut-top-k", type=int, default=20)
    ap.add_argument("--shortcut-max-paths", type=int, default=2_000_000)
    ap.add_argument("--max-queries", type=int, default=0, help="keyed sample of valid/test queries (0 = all)")
    ap.add_argument("--seed", type=int, default=1566911444)
    ap.add_argument("--num-negatives", type=int, default=50)
    ap.add_argument("--source", default="")
    args = ap.parse_args()

    out = os.path.join(args.data_root, args.dataset)
    entities, relations, train, valid, test = build_base(os.path.join(out, "raw"))
    n_valid, n_test = len(valid), len(test)
    valid, valid_rest = sample_queries(valid, args.max_queries, args.seed, "valid")
    test, test_rest = sample_queries(test, args.max_queries, args.seed, "test")
    heldout_all = np.concatenate([valid, valid_rest, test, test_rest])
    base = write_base(out, entities, relations, train, valid, test,
                      {"dataset": args.dataset, "source": args.source,
                       "full_valid_queries": n_valid, "full_test_queries": n_test,
                       "query_sample": args.max_queries or None},
                      num_negatives=args.num_negatives, neg_seed=args.seed,
                      filter_extra=np.concatenate([valid_rest, test_rest]))
    print("base:", {k: base[k] for k in ("num_entities", "num_relations", "num_train", "num_valid", "num_test",
                                        "num_filter_extra")})

    custom = {}
    for spec in args.reify_set:
        name, rels = spec.split("=", 1)
        custom[name] = rels.split(",")
    index = None
    common = {"dataset": args.dataset, "built_from": "official train triples only"}
    for v in args.variants:
        if v == "original":
            res = (entities, relations, train, {"num_added_relations": 0, "num_added_entities": 0,
                                                "num_added_edges": 0, "unanswerable_query_relations": []})
        elif v == "inverse":
            res = variant_inverse(entities, relations, train)
        elif v.startswith("reified_"):
            chosen, rule = choose_reify_set(v[len("reified_"):], relations, train, test, args.reify_top_k, custom)
            res = variant_reified(entities, relations, train, test, chosen, rule)
        elif v.startswith("shortcut2_c"):
            index = index or ShortcutIndex(train)
            res = variant_shortcut2(entities, relations, train, heldout_all, index, args.shortcut_top_k,
                                    parse_cap(v[len("shortcut2_c"):]), args.shortcut_max_paths, args.seed)
        else:
            raise ValueError(v)
        ents, rels, graph, info = res
        m = write_variant(out, v, ents, rels, graph, dict(common, transformation=v, **info))
        extra = f", {100 * info['share_of_test_queries_unanswerable']:.1f}% test queries unanswerable" \
            if "share_of_test_queries_unanswerable" in info else ""
        print(f"{v}: {m['num_graph_edges']} edges, {m['num_relations']} relations, {m['num_entities']} entities{extra}")


if __name__ == "__main__":
    main()
