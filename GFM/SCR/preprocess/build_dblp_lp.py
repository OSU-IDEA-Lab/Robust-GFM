"""Export original DBLP paper-venue link prediction (v1-v3) for SCR.

Source: MAGNN bundles DBLP_lp_pc_var{1,2,3}_train_pc. The variants differ only in
where the research-area node attaches:

    v1  paper  - area      v2  venue - area      v3  author - area

Author-paper, paper-term, and the train-only paper-venue edges are identical
in all three. Facts are written forward only; SCR adds inverses. Held-out
paper-venue pairs are not in the message graph. negatives.npz keeps the
bundle's 19 negatives per paper so a run can rank the true venue plus 3 of them.
"""

import argparse
import json
import os
import shutil

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
AUTHOR, PAPER, TERM, CONF, AREA = range(5)
PREFIX = {AUTHOR: "a", PAPER: "p", TERM: "t", CONF: "c", AREA: "r"}
RELATIONS = {
    (PAPER, AUTHOR): "paper_has_author",
    (PAPER, TERM): "paper_has_term",
    (PAPER, CONF): "published_in",
    (PAPER, AREA): "paper_in_area",
    (CONF, AREA): "venue_in_area",
    (AUTHOR, AREA): "author_in_area",
}
SHARED = ("paper_has_author", "paper_has_term", "published_in")


def load_bundle(root, variant):
    d = os.path.join(root, "DBLP_lp_pc_var{}_train_pc".format(variant[1:]))
    types = np.load(os.path.join(d, "node_types.npy")).astype(np.int64)
    adj = sp.load_npz(os.path.join(d, "adjM.npz")).tocoo()
    pos = np.load(os.path.join(d, "train_val_test_pos_paper_conf.npz"))
    neg = np.load(os.path.join(d, "train_val_test_neg_paper_conf.npz"))
    return d, types, adj, {k: pos[k] for k in pos.files}, {k: neg[k] for k in neg.files}


def facts_of(types, adj):
    offsets = np.r_[0, np.cumsum(np.bincount(types))]
    local = np.arange(len(types)) - offsets[types]
    name = lambda n: f"{PREFIX[int(types[n])]}{int(local[n])}"  # noqa: E731
    facts = {}
    for h, t in zip(adj.row, adj.col):
        key = (int(types[h]), int(types[t]))
        if key in RELATIONS:
            facts.setdefault(RELATIONS[key], set()).add((name(h), name(t)))
    return {r: sorted(e) for r, e in facts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnn-root", required=True, help="directory of MAGNN preprocessed bundles")
    ap.add_argument("--out-root", default=os.path.join(SCR, "kg-datasets", "FoundationNodeKG"))
    ap.add_argument("--variants", nargs="+", default=["v1", "v2", "v3"])
    args = ap.parse_args()

    reference = None
    for v in args.variants:
        src, types, adj, pos, neg = load_bundle(args.magnn_root, v)
        facts = facts_of(types, adj)
        shared = {r: facts[r] for r in SHARED}
        if reference is None:
            reference = dict(types=types, shared=shared, pos=pos, neg=neg)
        else:
            assert np.array_equal(types, reference["types"]), f"{v}: node types differ"
            assert shared == reference["shared"], f"{v}: author/term/venue facts differ"
            for k in pos:
                assert np.array_equal(pos[k], reference["pos"][k]), f"{v}: {k} differs"
            for k in neg:
                assert np.array_equal(neg[k], reference["neg"][k]), f"{v}: {k} differs"

        train_pc = {(f"p{p}", f"c{c}") for p, c in pos["train_pos"]}
        assert set(facts["published_in"]) == train_pc, f"{v}: graph venue edges != train positives"
        held_out = {(f"p{p}", f"c{c}") for s in ("val_pos", "test_pos") for p, c in pos[s]}
        assert not held_out & set(facts["published_in"]), f"{v}: held-out venue edge in the graph"
        for s in ("val", "test"):
            per_paper = np.bincount(neg[f"{s}_neg"][:, 0])
            assert set(per_paper[pos[f"{s}_pos"][:, 0]]) == {19}, f"{v}: {s} is not a 20-way ranking"

        out = os.path.join(args.out_root, f"DBLP_LP_{v}")
        if os.path.isdir(out):
            shutil.rmtree(out)
        raw = os.path.join(out, "raw")
        os.makedirs(raw)
        graph = [(h, r, t) for r in sorted(facts) if r != "published_in" for h, t in facts[r]]
        graph += [(h, "published_in", t) for h, t in facts["published_in"]]
        files = {
            "train": graph,
            "valid": [(f"p{p}", "published_in", f"c{c}") for p, c in pos["val_pos"]],
            "test": [(f"p{p}", "published_in", f"c{c}") for p, c in pos["test_pos"]],
        }
        for fname, lines in files.items():
            with open(os.path.join(raw, f"{fname}.txt"), "w") as f:
                f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in lines)
        np.savez(os.path.join(raw, "negatives.npz"), val_neg=neg["val_neg"], test_neg=neg["test_neg"])

        counts = {r: len(e) for r, e in facts.items()}
        meta = dict(variant=v, arm="original", source=src, num_nodes=int(len(types)),
                    nodes_per_type=dict(zip(["author", "paper", "term", "venue", "area"],
                                            np.bincount(types).tolist())),
                    relation_counts=counts, num_graph_facts=len(graph),
                    num_valid=len(files["valid"]), num_test=len(files["test"]), candidates_per_query=20)
        with open(os.path.join(raw, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  DBLP_LP_{v}: {len(types)} nodes, {len(graph)} facts {counts}")


if __name__ == "__main__":
    main()
