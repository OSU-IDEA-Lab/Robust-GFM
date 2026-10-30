#!/usr/bin/env python3
"""Build the IMDb link-prediction split archives IMDB_{md,ml}_shared_splits.npz.

  * movie frame  = `read_imdb_frame` from build_IMDB_my_shared_splits.py (dedup on movie_imdb_link,
    drop rows without actor_1_name / director_name, keep Action/Comedy/Drama -> 4,180 movies);
  * partition    = nested movie split, the same scheme as the DBLP builder:
    train_test_split(movies, test_size=0.3, random_state=1566911444), then
    train_test_split(rest, test_size=2/3, random_state=1566911444) -> 2,926 / 418 / 836,
    each split sorted by movie index;
  * md positives = (movie, director), director = index in the sorted list of unique director
    names (2,081); ml positives = (movie, movie) (the Link node is a copy of the movie).

Expected sha256: IMDB_ml_shared_splits.npz 65b7013f...; IMDB_md_shared_splits.npz stores the same
arrays as the distributed file (765b6588...), although the zip container bytes can differ.
IMDB_my_shared_splits.npz is built afterwards by build_IMDB_my_shared_splits.py, which inherits
this movie partition.
"""
import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_IMDB_my_shared_splits import read_imdb_frame  # noqa: E402

SEED = 1566911444
_DATA = os.environ.get(
    "INV_GNN_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(_DATA, "raw", "IMDB", "movie_metadata.csv"))
    ap.add_argument("--out-dir", default=os.path.join(_DATA, "preprocessed", "IMDB_shared_splits"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    movies = read_imdb_frame(args.csv)
    n = len(movies)
    idx = np.arange(n)
    train, rest = train_test_split(idx, test_size=0.3, random_state=args.seed)
    val, test = train_test_split(rest, test_size=2 / 3, random_state=args.seed)
    parts = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}

    directors = sorted(set(movies["director_name"].dropna()))
    d_map = {name: i for i, name in enumerate(directors)}

    md = {
        f"{s}_pos": np.array([[int(i), d_map[movies.iloc[i]["director_name"]]] for i in a], dtype=np.int64)
        for s, a in parts.items()
    }
    ml = {f"{s}_pos": np.stack([a, a], axis=1).astype(np.int64) for s, a in parts.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "IMDB_md_shared_splits.npz")
    ml_path = os.path.join(args.out_dir, "IMDB_ml_shared_splits.npz")
    np.savez(md_path, **md, seed=np.int64(args.seed), num_movies=np.int64(n),
             num_directors=np.int64(len(directors)))
    np.savez(ml_path, **ml, seed=np.int64(args.seed), num_movies=np.int64(n))
    print(f"Saved {md_path} and {ml_path}: train={len(parts['train'])} val={len(parts['val'])} "
          f"test={len(parts['test'])} movies={n} directors={len(directors)}")


if __name__ == "__main__":
    main()
