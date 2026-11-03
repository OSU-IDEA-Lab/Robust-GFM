"""Export original IMDb movie-director link prediction (v1, v3) for SCR.

Source: MAGNN bundles IMDB_rgcn_lp_md_v1 and IMDB_rgcn_lp_md_v3. Actor edges
attach to the movie in v1 and to the link node in v3. Both keep movie-link
edges and training-only movie-director edges. Held-out director edges are the
valid and test queries. candidates.txt is every director. negatives.npz is the
bundle's 3 fixed negatives per held-out movie.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
TYPES = ("movie", "director", "actor", "link")
PREFIX = {"movie": "m", "director": "d", "actor": "a", "link": "l"}
BUNDLE_REL = {"movie-actor": "movie_to_actor", "movie-link": "movie_to_link", "link-actor": "link_to_actor",
              "movie-director": "movie_to_director", "link-director": "link_to_director"}
FORWARD = ("movie_to_actor", "movie_to_link", "link_to_actor", "link_to_director")
TARGET = "movie_to_director"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnn-root", required=True)
    ap.add_argument("--out-root", default=os.path.join(SCR, "kg-datasets", "FoundationNodeKG"))
    ap.add_argument("--variants", nargs="+", default=["v1", "v3"])
    args = ap.parse_args()

    reference = None
    for v in args.variants:
        src = os.path.join(args.magnn_root, f"IMDB_rgcn_lp_md_{v}")
        graph = torch.load(os.path.join(src, "graph_data.pt"), weights_only=False)
        meta = torch.load(os.path.join(src, "meta.pt"), weights_only=False)
        num_nodes = meta["num_nodes"]
        splits = {k: np.asarray(val, dtype=np.int64) for k, val in meta["splits"].items()}
        if reference is None:
            reference = dict(num_nodes=num_nodes, splits=splits)
        assert num_nodes == reference["num_nodes"], f"{v}: node counts differ"
        for k in splits:
            assert np.array_equal(splits[k], reference["splits"][k]), f"{v}: {k} differs"

        offset, cur = {}, 0
        for t in TYPES:
            offset[t], cur = cur, cur + num_nodes[t]

        def name(n):
            for t in TYPES:
                if offset[t] <= n < offset[t] + num_nodes[t]:
                    return f"{PREFIX[t]}{n - offset[t]}"
            raise KeyError(n)

        raw = defaultdict(set)
        for key, (u, w) in graph.items():
            rel = BUNDLE_REL[key]
            head, tail = rel.split("_to_")
            for a, b in zip(np.asarray(u).tolist(), np.asarray(w).tolist()):
                raw[rel].add((offset[head] + a, offset[tail] + b))
        train_md = {(m, offset["director"] + d) for m, d in splits["train_pos"].tolist()}
        assert raw[TARGET] == train_md, f"{v}: graph movie-director edges != training positives"
        held_out = {(m, offset["director"] + d) for k in ("val_pos", "test_pos") for m, d in splits[k].tolist()}
        assert not held_out & raw[TARGET], f"{v}: held-out movie-director edge in the graph"

        directors = [f"d{i}" for i in range(num_nodes["director"])]
        facts = [(name(h), r, name(t)) for r in FORWARD for h, t in sorted(raw.get(r, ()))]
        facts += [(name(h), TARGET, name(t)) for h, t in sorted(raw[TARGET])]
        files = {
            "train": facts,
            "valid": [(f"m{m}", TARGET, f"d{d}") for m, d in splits["val_pos"].tolist()],
            "test": [(f"m{m}", TARGET, f"d{d}") for m, d in splits["test_pos"].tolist()],
        }
        out = os.path.join(args.out_root, f"IMDB_MD_{v}")
        if os.path.isdir(out):
            shutil.rmtree(out)
        raw_dir = os.path.join(out, "raw")
        os.makedirs(raw_dir)
        for fname, lines in files.items():
            with open(os.path.join(raw_dir, f"{fname}.txt"), "w") as f:
                f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in lines)
        with open(os.path.join(raw_dir, "candidates.txt"), "w") as f:
            f.writelines(f"{d}\n" for d in directors)
        flat = lambda pos, neg: np.c_[np.repeat(pos[:, 0], neg.shape[1]), neg.ravel()]  # noqa: E731
        np.savez(os.path.join(raw_dir, "negatives.npz"),
                 val_neg=flat(splits["val_pos"], splits["val_neg"]),
                 test_neg=flat(splits["test_pos"], splits["test_neg"]))
        counts = defaultdict(int)
        for _, r, _ in facts:
            counts[r] += 1
        info = dict(variant=v, arm="original", task="movie_director", source=src,
                    num_nodes=int(sum(num_nodes.values())), nodes_per_type=dict(num_nodes),
                    relation_counts=dict(counts), num_graph_facts=len(facts),
                    num_valid=len(files["valid"]), num_test=len(files["test"]),
                    candidates_per_query=len(directors),
                    fixed_negatives_per_query=int(splits["test_neg"].shape[1]))
        with open(os.path.join(raw_dir, "meta.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  IMDB_MD_{v}: {len(facts)} facts {dict(counts)}")


if __name__ == "__main__":
    main()
