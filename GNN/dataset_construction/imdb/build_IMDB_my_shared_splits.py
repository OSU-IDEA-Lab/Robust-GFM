"""Shared movie->year LP splits for IMDB.

Unlike the md/ml builders this does NOT draw a new partition. It INHERITS the
canonical movie partition from an existing shared-split archive, so Movie-Year,
Movie-Director and Movie-Link hold out exactly the same movies and their numbers
are directly comparable (IMDB_md_shared_splits.npz and IMDB_ml_shared_splits.npz
induce the same 2926/418/836 movie partition).

3 of the 4180 movies have no ``title_year``. They keep their nodes and every other
edge; they are simply absent from the target pairs, giving 2925/418/834.
"""

import argparse

import numpy as np
import pandas as pd


def read_imdb_frame(csv_path):
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
    return movies[labels != -1].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to movie_metadata.csv")
    parser.add_argument(
        "--inherit-from",
        required=True,
        help="Canonical archive whose movie partition is reused (IMDB_md_shared_splits.npz)",
    )
    parser.add_argument("--out", default="IMDB_my_shared_splits.npz")
    args = parser.parse_args()

    movies = read_imdb_frame(args.csv)
    n = len(movies)

    years = sorted({int(y) for y in movies["title_year"].dropna()})
    y_map = {y: i for i, y in enumerate(years)}
    year_of = {
        i: y_map[int(v)]
        for i, v in movies["title_year"].items()
        if pd.notna(v)
    }

    src = np.load(args.inherit_from)
    seed = int(src["seed"]) if "seed" in src.files else -1
    if "num_movies" in src.files and int(src["num_movies"]) != n:
        raise SystemExit(
            f"{args.inherit_from} was built on {int(src['num_movies'])} movies, "
            f"this frame has {n}; the CSV filter does not match."
        )

    out, dropped = {}, 0
    for split in ("train", "val", "test"):
        heads = src[f"{split}_pos"][:, 0].astype(int)
        if len(set(heads.tolist())) != len(heads):
            raise SystemExit(f"{args.inherit_from} repeats a movie in {split}")
        rows = [[int(i), year_of[int(i)]] for i in heads if int(i) in year_of]
        dropped += len(heads) - len(rows)
        out[f"{split}_pos"] = (
            np.array(rows, dtype=np.int64) if rows else np.empty((0, 2), dtype=np.int64)
        )

    np.savez(
        args.out,
        seed=np.int64(seed),
        num_movies=np.int64(n),
        num_years=np.int64(len(years)),
        year_values=np.array(years, dtype=np.int64),
        inherited_from=np.array(args.inherit_from),
        **out,
    )
    print(
        f"Saved {args.out}  train={len(out['train_pos'])} val={len(out['val_pos'])} "
        f"test={len(out['test_pos'])}  years={len(years)}  "
        f"dropped_no_year={dropped}  partition inherited from {args.inherit_from}"
    )


if __name__ == "__main__":
    main()
