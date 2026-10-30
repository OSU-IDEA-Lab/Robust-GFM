"""Export the WN18RR variants of the GNN experiments into the shared format (see kg_format.py).

Source (read only): GNN/data/wordnet_3hops_augmented_full/, built from the official train.txt only
(GNN/docs/DATASET_CONSTRUCTION_FREEBASE_WORDNET.md).

    original_splits/{entities,relations}.dict, train/valid/test.txt   -> base/
    <variant>/data.txt (message graph), relations.dict                -> variants/<variant>/

Variants: no_changes (official train graph), all_inverse_edges (+ named inverse edges),
transitive_edges (+ 2/3-hop shortcut relations), universal_edges (the union graph).
No entities are added by any WN18RR variant.
"""

import argparse
import os

from kg_format import read_lines, write_base, write_variant

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("GFM_DATA", os.path.join(HERE, "..", "data"))
DEFAULT_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "GNN", "data", "wordnet_3hops_augmented_full"))
VARIANTS = ["no_changes", "all_inverse_edges", "transitive_edges", "universal_edges"]


def read_dict(path):
    pairs = [line.split("\t") for line in read_lines(path) if line]
    pairs.sort(key=lambda p: int(p[0]))
    assert [int(p[0]) for p in pairs] == list(range(len(pairs))), f"non-contiguous ids in {path}"
    return [p[1] for p in pairs]


def read_named_triples(path):
    return [tuple(line.split("\t")) for line in read_lines(path) if line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "WN18RR"))
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--num-negatives", type=int, default=50)
    args = ap.parse_args()

    splits = os.path.join(args.src, "original_splits")
    entities = read_dict(os.path.join(splits, "entities.dict"))
    relations = read_dict(os.path.join(splits, "relations.dict"))
    eid = {e: i for i, e in enumerate(entities)}
    rid = {r: i for i, r in enumerate(relations)}

    def ids(named):
        return [(eid[h], rid[r], eid[t]) for h, r, t in named]

    train_named = read_named_triples(os.path.join(splits, "train.txt"))
    base = write_base(
        args.out, entities, relations,
        ids(train_named),
        ids(read_named_triples(os.path.join(splits, "valid.txt"))),
        ids(read_named_triples(os.path.join(splits, "test.txt"))),
        {"dataset": "WN18RR", "source": os.path.relpath(args.src, HERE)},
        num_negatives=args.num_negatives,
    )
    print("base:", {k: base[k] for k in ("num_entities", "num_relations", "num_train", "num_valid", "num_test")})

    train_set = set(train_named)
    for v in args.variants:
        vdir = os.path.join(args.src, v)
        v_entities = read_dict(os.path.join(vdir, "entities.dict"))
        assert v_entities == entities, f"{v}: entity vocabulary differs from the official one"
        v_rel_dict = read_dict(os.path.join(vdir, "relations.dict"))
        added = [r for r in v_rel_dict if r not in rid]
        v_relations = relations + added
        v_rid = {r: i for i, r in enumerate(v_relations)}
        named = read_named_triples(os.path.join(vdir, "data.txt"))
        graph = [(eid[h], v_rid[r], eid[t]) for h, r, t in named]
        base_edges = sum(1 for h, r, t in named if r in rid)
        base_subset = {x for x in named if x[1] in rid}
        manifest = {
            "dataset": "WN18RR",
            "source_variant_dir": os.path.relpath(vdir, HERE),
            "transformation": v,
            "num_added_relations": len(added),
            "num_added_entities": 0,
            "num_base_relation_edges": base_edges,
            "num_added_relation_edges": len(named) - base_edges,
            "base_relation_edges_equal_official_train": base_subset == train_set,
            "unanswerable_query_relations": [],
        }
        m = write_variant(args.out, v, v_entities, v_relations, graph, manifest)
        print(f"{v}: {m['num_graph_edges']} edges, {m['num_relations']} relations "
              f"(+{len(added)}), base edges == official train: {manifest['base_relation_edges_equal_official_train']}")


if __name__ == "__main__":
    main()
