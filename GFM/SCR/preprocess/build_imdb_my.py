"""Export original IMDb movie-year link prediction (v1-v4) for SCR.

The query is (movie, movie_has_year, year). Structure is the original
node-classification graph IMDB_rgcn_v1 .. v4, read only. Year edges are added
for training movies and attached to the movie, never to the link node.
Validation and test years stay out of the graph. candidates.txt lists every
observed year.

The movie split is the shared movie-link split, where the movie id equals the
link id. Movies with no title_year are dropped from the queries.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
PREFIX = {"movie": "m", "director": "d", "actor": "a", "link": "l"}
TARGET = "movie_has_year"


def movie_years(csv_path):
    movies = (
        pd.read_csv(csv_path, encoding="utf-8")
        .drop_duplicates(subset=["movie_imdb_link"])
        .dropna(axis=0, subset=["actor_1_name", "director_name"])
        .reset_index(drop=True)
    )
    labels = np.full(len(movies), -1, dtype=int)
    for idx, genres in movies["genres"].items():
        for g in str(genres).split("|"):
            if g == "Action":
                labels[idx] = 0
                break
            elif g == "Comedy":
                labels[idx] = 1
                break
            elif g == "Drama":
                labels[idx] = 2
                break
    movies = movies[labels != -1].reset_index(drop=True)
    raw = movies["title_year"].to_numpy()
    present = ~pd.isna(raw)
    years = sorted({int(y) for y in raw[present]})
    yid = {y: i for i, y in enumerate(years)}
    assigned = np.full(len(movies), -1, dtype=np.int64)
    assigned[present] = [yid[int(y)] for y in raw[present]]
    return assigned, years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnn-root", required=True)
    ap.add_argument("--csv", required=True, help="IMDb movie_metadata.csv")
    ap.add_argument("--split", required=True, help="IMDB_ml_shared_splits.npz with train_pos/val_pos/test_pos")
    ap.add_argument("--out-root", default=os.path.join(SCR, "kg-datasets", "FoundationNodeKG"))
    ap.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4"])
    args = ap.parse_args()

    year_of, years = movie_years(args.csv)
    z = np.load(args.split)
    splits = {k: np.asarray(z[k], dtype=np.int64) for k in ("train_pos", "val_pos", "test_pos")}
    queries = {}
    for name, pairs in (("train", splits["train_pos"]), ("valid", splits["val_pos"]), ("test", splits["test_pos"])):
        rows = []
        for movie, link in pairs.tolist():
            assert movie == link
            if year_of[movie] >= 0:
                rows.append((movie, int(year_of[movie])))
        queries[name] = rows
    held = {(m, y) for split in ("valid", "test") for m, y in queries[split]}

    for v in args.variants:
        graph = torch.load(os.path.join(args.magnn_root, f"IMDB_rgcn_{v}", "graph_data.pt"), weights_only=False)
        facts = []
        for key, (src_idx, dst_idx) in graph.items():
            head, tail = key.split("-")
            rel = f"{head}_to_{tail}"
            for a, b in zip(np.asarray(src_idx).tolist(), np.asarray(dst_idx).tolist()):
                facts.append((f"{PREFIX[head]}{a}", rel, f"{PREFIX[tail]}{b}"))
        year_edges = {(m, y) for m, y in queries["train"]}
        assert not held & year_edges

        files = {
            "train": facts + [(f"m{m}", TARGET, f"y{y}") for m, y in queries["train"]],
            "valid": [(f"m{m}", TARGET, f"y{y}") for m, y in queries["valid"]],
            "test": [(f"m{m}", TARGET, f"y{y}") for m, y in queries["test"]],
        }
        out = os.path.join(args.out_root, f"IMDB_MY_{v}")
        if os.path.isdir(out):
            shutil.rmtree(out)
        raw = os.path.join(out, "raw")
        os.makedirs(raw)
        for fname, lines in files.items():
            with open(os.path.join(raw, f"{fname}.txt"), "w") as f:
                f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in lines)
        with open(os.path.join(raw, "candidates.txt"), "w") as f:
            f.writelines(f"y{i}\n" for i in range(len(years)))
        with open(os.path.join(raw, "year_names.txt"), "w") as f:
            f.writelines(f"{i}\t{y}\n" for i, y in enumerate(years))
        counts = defaultdict(int)
        for _, r, _ in facts:
            counts[r] += 1
        info = dict(variant=v, arm="original", task="movie_year", source=f"IMDB_rgcn_{v}",
                    num_graph_facts=len(facts), relation_counts=dict(counts),
                    num_years=len(years), dropped_missing_year=int((year_of < 0).sum()),
                    num_train=len(queries["train"]), num_valid=len(queries["valid"]),
                    num_test=len(queries["test"]))
        with open(os.path.join(raw, "meta.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  IMDB_MY_{v}: {len(facts)} graph facts, years {len(years)}, "
              f"train/valid/test {len(queries['train'])}/{len(queries['valid'])}/{len(queries['test'])}")


if __name__ == "__main__":
    main()
