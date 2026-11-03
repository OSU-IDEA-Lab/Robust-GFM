"""Export original IMDb node classification (v1-v4) as label-entity triples for SCR.

The four graphs are the same movies, directors, actors, and link nodes. Director
and actor edges attach to the movie or to its link node:

    v1  movie-director, movie-actor, movie-link
    v2  link-director,  link-actor,  movie-link
    v3  movie-director, link-actor,  movie-link
    v4  link-director,  movie-actor, movie-link

Labels, features, and the train/valid/test movies come from the MAGNN source
bundle IMDB_preprocessed_star. Genre edges are written only for training movies.
Held-out genre edges are the valid and test queries and are not in the graph.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
MOVIE, DIRECTOR, ACTOR, LINK = 0, 1, 2, 3
PREFIX = {MOVIE: "m", DIRECTOR: "d", ACTOR: "a", LINK: "l"}
FORWARD = ("movie_to_director", "movie_to_actor", "movie_to_link", "link_to_director", "link_to_actor")
PRESENT = {
    "v1": ("movie_to_director", "movie_to_actor"),
    "v2": ("link_to_director", "link_to_actor"),
    "v3": ("movie_to_director", "link_to_actor"),
    "v4": ("link_to_director", "movie_to_actor"),
}
LABEL_REL = "has_genre"


def typed_pairs(adj, type_mask, src_t, dst_t):
    mask = (type_mask[adj.row] == src_t) & (type_mask[adj.col] == dst_t)
    pairs = np.stack([adj.row[mask], adj.col[mask]], axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def features_of(prefix, type_mask):
    blocks = [sp.load_npz(os.path.join(prefix, f"features_{t}.npz")).toarray().astype(np.float32)
              for t in range(int(type_mask.max()) + 1)]
    dim = max(block.shape[1] for block in blocks)
    features = np.zeros((len(type_mask), dim), dtype=np.float32)
    for t, block in enumerate(blocks):
        if block.shape[1] < dim:
            block = np.concatenate([block, np.zeros((block.shape[0], dim - block.shape[1]), np.float32)], axis=1)
        features[type_mask == t] = block
    return features


def name(node, type_mask):
    return f"{PREFIX[int(type_mask[node])]}{int(node)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnn-root", required=True)
    ap.add_argument("--source", default="IMDB_preprocessed_star")
    ap.add_argument("--out-root", default=os.path.join(SCR, "kg-datasets", "FoundationNodeKG"))
    ap.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4"])
    args = ap.parse_args()

    prefix = os.path.join(args.magnn_root, args.source)
    type_mask = np.load(os.path.join(prefix, "node_types.npy")).astype(np.int64)
    labels = np.load(os.path.join(prefix, "labels.npy")).astype(np.int64)
    splits = np.load(os.path.join(prefix, "train_val_test_idx.npz"))
    features = features_of(prefix, type_mask)
    adj = sp.load_npz(os.path.join(prefix, "adjM.npz")).tocoo()
    pairs = dict(
        movie_director=typed_pairs(adj, type_mask, MOVIE, DIRECTOR),
        movie_actor=typed_pairs(adj, type_mask, MOVIE, ACTOR),
        movie_link=typed_pairs(adj, type_mask, MOVIE, LINK),
    )
    split_idx = {"train": splits["train_idx"], "val": splits["val_idx"], "test": splits["test_idx"]}

    for variant in args.variants:
        link_of = {int(m): int(link) for m, link in pairs["movie_link"]}
        raw = defaultdict(set)

        def add(rel, src, dst):
            raw[rel].add((int(src), int(dst)))

        for movie, link in pairs["movie_link"]:
            add("movie_to_link", movie, link)
        attach = PRESENT[variant]
        for movie, director in pairs["movie_director"]:
            rel = "movie_to_director" if "movie_to_director" in attach else "link_to_director"
            add(rel, movie if rel.startswith("movie") else link_of[int(movie)], director)
        for movie, actor in pairs["movie_actor"]:
            rel = "movie_to_actor" if "movie_to_actor" in attach else "link_to_actor"
            add(rel, movie if rel.startswith("movie") else link_of[int(movie)], actor)
        graph = [(name(h, type_mask), r, name(t, type_mask)) for r in FORWARD for h, t in sorted(raw.get(r, ()))]
        label_rows = {
            s: [(name(int(m), type_mask), LABEL_REL, f"genre{int(labels[int(m)])}") for m in split_idx[s]]
            for s in ("train", "val", "test")
        }
        files = {"train": graph + label_rows["train"], "valid": label_rows["val"], "test": label_rows["test"]}

        out = os.path.join(args.out_root, f"IMDB_NC_{variant}")
        if os.path.isdir(out):
            shutil.rmtree(out)
        raw_dir = os.path.join(out, "raw")
        os.makedirs(raw_dir)
        for fname, lines in files.items():
            with open(os.path.join(raw_dir, f"{fname}.txt"), "w") as f:
                f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in lines)

        vocab = {}
        for fname in ("train", "valid", "test"):
            for h, _, t in files[fname]:
                for e in (h, t):
                    vocab.setdefault(e, len(vocab))
        feats = np.zeros((len(vocab), features.shape[1]), dtype=np.float32)
        for ent, idx in vocab.items():
            if not ent.startswith("genre"):
                feats[idx] = features[int(ent[1:])]
        torch.save(torch.from_numpy(feats), os.path.join(raw_dir, "node_feats.pt"))

        counts = {}
        for _, r, _ in graph:
            counts[r] = counts.get(r, 0) + 1
        meta = dict(variant=variant, arm="original", source=prefix, num_entities=len(vocab),
                    num_graph_facts=len(graph), relation_counts=counts,
                    num_train_labels=len(label_rows["train"]), num_valid=len(label_rows["val"]),
                    num_test=len(label_rows["test"]))
        with open(os.path.join(raw_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  IMDB_NC_{variant}: {len(vocab)} entities, {len(graph)} facts {counts}")


if __name__ == "__main__":
    main()
