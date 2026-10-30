#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # common/: shared canonical IMDb features
from inv_rgcn.data import ConditionalGraphData, canonicalize_edges, save_graph_data
from inv_rgcn.utils import parse_str_list, set_deterministic
from imdb_canonical_features import (
    CANONICAL_FEATURE_RULE,
    build_canonical_features,
    feature_table_hash,
    frame_membership,
)

SEED = 1566911444
VARIANTS = ("v1", "v2", "v3", "v4", "universal")
# 'universal' is the union of a task's variants, compiled through this same
# pipeline so the universal-graph arm can be trained by the SAME runner and
# architecture as the invariant arm. Without it the universal point comes from a
# different codebase and any invariant-vs-universal gap mixes method with
# architecture.
TASKS = ("nc", "movie_link", "movie_director", "movie_year")

# Year relations are appended LAST so every pre-existing relation id is unchanged.
REL_NAMES = [
    "movie_to_director", "director_to_movie",
    "movie_to_actor", "actor_to_movie",
    "movie_to_link", "link_to_movie",
    "link_to_director", "director_to_link",
    "link_to_actor", "actor_to_link",
    "movie_to_year", "year_to_movie",
    "link_to_year", "year_to_link",
]
REL = {name: i for i, name in enumerate(REL_NAMES)}
# Year is node type 4, appended last for the same reason.
NODE_TYPES = {"movie": 0, "director": 1, "actor": 2, "link": 3, "year": 4}

# Tasks that carry a ``year`` node type at all. ``movie_link`` is deliberately
# excluded: it is kept frozen as the benchmark-contamination case study,
# so its graphs must not change.
YEAR_TASKS = ("nc", "movie_director", "movie_year")

# Which endpoint(s) each RELOCATABLE attribute attaches to, per task and variant.
#
# The relocated pair is always the two attributes that are NOT a target endpoint,
# which is what gives every task a complete 2x2 family. A target relation never
# appears here -- it is emitted train-only by build_physical.
#
# Slot convention (unchanged from the historical builds and SlotGAT's
# VARIANT_EDGES table): v1 = both on movie, v2 = both on link, v3/v4 = one each
# way. ``universal`` places every relocatable attribute on BOTH endpoints.
PLACEMENT: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    # NC has no held-out edge, so {director, actor} relocate exactly as before and
    # year enters as FIXED context on the movie in every variant -- literally
    # "add year to movie". This preserves the four-variant family and the
    # IMDB_var1..5 naming every NC runner depends on.
    "nc": {
        "v1": {"director": ("movie",), "actor": ("movie",), "year": ("movie",)},
        "v2": {"director": ("link",), "actor": ("link",), "year": ("movie",)},
        "v3": {"director": ("movie",), "actor": ("link",), "year": ("movie",)},
        "v4": {"director": ("link",), "actor": ("movie",), "year": ("movie",)},
        # 'universal' (IMDB_var5) is the UNION OF THIS TASK'S OWN VARIANTS, not a
        # transformation variant. {director, actor} relocate, so the union carries them
        # on both endpoints; the year is fixed on the movie in every variant, so the
        # union carries it on the movie only.
        "universal": {"director": ("movie", "link"), "actor": ("movie", "link"),
                      "year": ("movie",)},
    },
    # Movie-Director keeps its historical TWO-variant family: the director is the
    # target endpoint and cannot move, and the ACTOR is the only attribute that
    # relocates. The year is FIXED CONTEXT on the movie, exactly as in nc -- "connect
    # year to movie". The four-variant LP family lives in movie_year, so there is no
    # need to stretch this one; v1/v3 are byte-identical either way (physical hashes
    # 98998bc09c36 / 069919789e03), so the only thing a wider family would change is
    # the universal arm.
    "movie_director": {
        "v1": {"actor": ("movie",), "year": ("movie",)},
        "v3": {"actor": ("link",), "year": ("movie",)},
        # The union of v1 and v3: the actor is the only attribute that relocates, so
        # it lands on both endpoints, while the year stays on the movie.
        "universal": {"actor": ("movie", "link"), "year": ("movie",)},
    },
    # Movie-Year: the year is the target endpoint, so the relocated pair is
    # {director, actor} -- i.e. the IMDb1-4 variant definitions verbatim,
    # with only the target swapped out from under them.
    "movie_year": {
        "v1": {"director": ("movie",), "actor": ("movie",)},
        "v2": {"director": ("link",), "actor": ("link",)},
        "v3": {"director": ("movie",), "actor": ("link",)},
        "v4": {"director": ("link",), "actor": ("movie",)},
        "universal": {"director": ("movie", "link"), "actor": ("movie", "link")},
    },
    # Frozen. Identical to movie_year's family, and with no year node at all.
    "movie_link": {
        "v1": {"director": ("movie",), "actor": ("movie",)},
        "v2": {"director": ("link",), "actor": ("link",)},
        "v3": {"director": ("movie",), "actor": ("link",)},
        "v4": {"director": ("link",), "actor": ("movie",)},
        "universal": {"director": ("movie", "link"), "actor": ("movie", "link")},
    },
}


def load_movies(raw_csv: str, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Use the exact row filtering/order used by the repository skip LP code.

    * Movie-Link (``ml``) matches ``read_imdb_frame_ml``.
    * Movie-Director (``md``) matches ``read_imdb_frame_md_mg``.
    * Movie-Year and node classification keep the same labeled IMDb subset.

    The two filters are measurably row-identical (4180 movies in the same order),
    which is what lets Movie-Year reuse the canonical movie partition verbatim.
    """
    movies = pd.read_csv(raw_csv, encoding="utf-8").drop_duplicates(subset=["movie_imdb_link"])
    if task == "movie_link":
        movies = movies.dropna(
            axis=0,
            subset=["movie_imdb_link", "actor_1_name", "director_name", "genres"],
        ).reset_index(drop=True)
    else:
        movies = movies.dropna(axis=0, subset=["actor_1_name", "director_name"]).reset_index(drop=True)

    labels = np.full(len(movies), -1, dtype=np.int64)
    for i, genres in movies["genres"].astype(str).items():
        for genre in genres.split("|"):
            genre = genre.strip()
            if genre == "Action":
                labels[i] = 0
                break
            if genre == "Comedy":
                labels[i] = 1
                break
            if genre == "Drama":
                labels[i] = 2
                break
    keep = np.where(labels >= 0)[0]
    return movies.iloc[keep].reset_index(drop=True), labels[keep]

def build_entities(movies: pd.DataFrame, task: str) -> dict:
    directors = sorted(set(movies["director_name"].dropna().tolist()))
    if task == "movie_link":
        # Exact CMPNN/skip ML schema: actor_1 only.
        actors = sorted(set(movies["actor_1_name"].dropna().tolist()))
    else:
        actors = sorted(set(
            movies["actor_1_name"].dropna().tolist()
            + movies["actor_2_name"].dropna().tolist()
            + movies["actor_3_name"].dropna().tolist()
        ))
    # Year nodes exist for every task except movie_link, which stays frozen.
    # They are appended AFTER the link block so off_movie/off_director/off_actor/
    # off_link are unchanged and the shared split archives keep working untouched.
    if task in YEAR_TASKS:
        years = sorted({int(y) for y in movies["title_year"].dropna().tolist()})
    else:
        years = []
    nm, nd, na, ny = len(movies), len(directors), len(actors), len(years)
    info = {
        "num_movies": nm,
        "num_directors": nd,
        "num_actors": na,
        "num_links": nm,
        "num_years": ny,
        "off_movie": 0,
        "off_director": nm,
        "off_actor": nm + nd,
        "off_link": nm + nd + na,
        "off_year": 2 * nm + nd + na,
        "num_nodes": 2 * nm + nd + na + ny,
        "director_to_idx": {x: i for i, x in enumerate(directors)},
        "actor_to_idx": {x: i for i, x in enumerate(actors)},
        "year_to_idx": {y: i for i, y in enumerate(years)},
    }
    node_type = np.empty(info["num_nodes"], dtype=np.int64)
    node_type[:nm] = NODE_TYPES["movie"]
    node_type[nm:nm + nd] = NODE_TYPES["director"]
    node_type[nm + nd:nm + nd + na] = NODE_TYPES["actor"]
    node_type[nm + nd + na:2 * nm + nd + na] = NODE_TYPES["link"]
    node_type[2 * nm + nd + na:] = NODE_TYPES["year"]
    info["node_type"] = node_type
    return info

def build_features_svd_movie_only(movies: pd.DataFrame, info: dict, dim: int, seed: int) -> np.ndarray:
    """LEGACY feature mode (``--feature-mode svd_movie_only``): 128-d TruncatedSVD of the
    movie BoW, unit-normalised, and ZERO rows for every other node type.  Kept only as an ablation; the default is
    the canonical pooled-BoW table shared with the other models (see
    ``build_features_canonical``)."""
    text = movies["plot_keywords"].fillna("").map(lambda x: str(x).replace("|", " "))
    try:
        sparse = CountVectorizer(min_df=2).fit_transform(text).astype(np.float32)
    except ValueError:
        sparse = CountVectorizer(min_df=1).fit_transform(text.map(lambda x: x or "emptytoken")).astype(np.float32)

    if sparse.shape[1] <= dim:
        movie_x = sparse.toarray().astype(np.float32)
    else:
        actual = max(1, min(dim, min(sparse.shape) - 1))
        movie_x = TruncatedSVD(n_components=actual, random_state=seed).fit_transform(sparse).astype(np.float32)
    if movie_x.shape[1] < dim:
        movie_x = np.pad(movie_x, ((0, 0), (0, dim - movie_x.shape[1])))

    # Only representation-independent target features are retained. Non-target node
    # features are zero, rather than being averaged through a variant-specific graph.
    x = np.zeros((info["num_nodes"], dim), dtype=np.float32)
    x[:info["num_movies"]] = movie_x[:, :dim]
    norm = np.linalg.norm(x[:info["num_movies"]], axis=1, keepdims=True)
    x[:info["num_movies"]] /= np.maximum(norm, 1e-12)
    return x


# Held-out target type per LP task: its pooled feature must be built from TRAINING
# pairs only, or a test movie's BoW would sit inside its true target's feature vector.
LP_TARGET_TYPE = {"movie_director": "director", "movie_year": "year", "movie_link": "link"}


def build_features_canonical(
    movies: pd.DataFrame, info: dict, task: str = "nc",
    train_target_pairs: "set[tuple[int, int]] | None" = None,
) -> np.ndarray:
    """The project-wide canonical IMDb node features (``imdb_canonical_features``):
    raw-count plot_keywords BoW on movies, and for director / actor / link / year the mean
    of their member movies' BoW, membership taken from the DATAFRAME so the table is
    identical in every variant.  Verified to reproduce
    ``preprocess_IMDB_rgcn_conditional.py``'s ``x.pt`` (the .6647 inputs) and SlotGAT's
    ``pooled_bow`` table exactly, up to each package's link-node id convention."""
    train_only = None
    if task in LP_TARGET_TYPE:
        if train_target_pairs is None:
            raise ValueError(f"{task}: canonical LP features need the training target pairs")
        ttype = LP_TARGET_TYPE[task]
        off = info[f"off_{ttype}"]
        # (movie index, local target index) -- pairs are stored as global node ids
        train_only = {ttype: [(int(m), int(t) - off) for m, t in train_target_pairs]}
    members = frame_membership(
        movies, info["director_to_idx"], info["actor_to_idx"],
        info.get("year_to_idx") if info.get("num_years") else None,
        train_only=train_only,
    )
    order = ["director", "actor", "link"] + (["year"] if info.get("num_years") else [])
    x = build_canonical_features(movies, members, order)
    if x.shape[0] != info["num_nodes"]:
        raise RuntimeError(f"canonical feature rows {x.shape[0]} != num_nodes {info['num_nodes']}")
    return x


def build_blocks(movies: pd.DataFrame, info: dict, task: str) -> list[dict]:
    dmap, amap = info["director_to_idx"], info["actor_to_idx"]
    blocks = []
    for movie, row in movies.iterrows():
        director = info["off_director"] + dmap[row["director_name"]]
        link = info["off_link"] + int(movie)
        actor_columns = ("actor_1_name",) if task == "movie_link" else (
            "actor_1_name", "actor_2_name", "actor_3_name"
        )
        all_actors = []
        for col in actor_columns:
            value = row.get(col)
            if pd.notna(value) and value in amap:
                all_actors.append(info["off_actor"] + amap[value])
        actor1 = info["off_actor"] + amap[row["actor_1_name"]]
        # 3 of 4180 movies have no title_year; they keep their node and every other
        # edge, and are simply absent from the Movie-Year target pairs.
        ymap = info.get("year_to_idx") or {}
        raw_year = row.get("title_year")
        year = (
            info["off_year"] + ymap[int(raw_year)]
            if ymap and pd.notna(raw_year) and int(raw_year) in ymap
            else None
        )
        blocks.append({
            "block_id": int(movie),
            "movie": int(movie),
            "director": int(director),
            "link": int(link),
            "actors": list(all_actors),
            "actor1": int(actor1),
            "year": year,
        })
    return blocks

def split_movie_ids(num_movies: int, seed: int) -> dict[str, np.ndarray]:
    if num_movies < 10:
        raise ValueError("IMDb node-classification preprocessing requires at least 10 movies")
    idx = np.arange(num_movies)
    train, val = train_test_split(idx, test_size=max(1, int(0.1 * len(idx))), random_state=seed)
    train, test = train_test_split(train, test_size=max(1, int(0.2 * len(idx))), random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


TARGET_KEY = {"movie_link": "link", "movie_director": "director", "movie_year": "year"}


def target_pairs(blocks: list[dict], task: str) -> np.ndarray:
    key = TARGET_KEY[task]
    return np.asarray(
        [(b["movie"], b[key]) for b in blocks if b[key] is not None], dtype=np.int64
    )


def sample_negative_tails(pos: np.ndarray, candidate_nodes: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in pos:
        true_by_head[int(head)].add(int(tail))
    candidates = np.asarray(candidate_nodes, dtype=np.int64)
    out = np.empty((len(pos), k), dtype=np.int64)
    for i, (head, _tail) in enumerate(pos):
        valid = np.asarray([x for x in candidates if int(x) not in true_by_head[int(head)]], dtype=np.int64)
        if len(valid) == 0:
            raise RuntimeError(f"No valid negative tails for head {head}")
        out[i] = rng.choice(valid, size=k, replace=len(valid) < k)
    return out


def _target_offset(task: str, info: dict) -> int:
    """Global-id offset of the block holding a task's target-type nodes."""
    return {
        "movie_link": info["off_link"],
        "movie_director": info["off_director"],
        "movie_year": info.get("off_year", 0),
    }[task]


def _convert_shared_pairs(arr: np.ndarray, task: str, info: dict) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected pair array [N,2], got {arr.shape}")
    out = arr.copy()
    offset = _target_offset(task, info)
    # Shared repository split files use local target-type ids.
    if len(out) and out[:, 1].max(initial=-1) < offset:
        out[:, 1] += offset
    return out


def _convert_shared_negatives(
    arr: np.ndarray,
    raw_pos: np.ndarray,
    task: str,
    info: dict,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64).copy()
    raw_pos = np.asarray(raw_pos, dtype=np.int64)
    offset = _target_offset(task, info)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D negatives, got {arr.shape}")

    # Tail-matrix format [Q,K] is identified by one row per positive query. A flat
    # pair format [Q*K,2] is identified otherwise. The only ambiguous case is K=2;
    # it is a flat pair list only when its first column exactly repeats query heads.
    is_flat_pairs = (
        arr.shape[1] == 2
        and (arr.shape[0] != raw_pos.shape[0] or np.array_equal(arr[:, 0], raw_pos[:, 0]))
    )
    if is_flat_pairs:
        if len(arr) and arr[:, 1].max(initial=-1) < offset:
            arr[:, 1] += offset
        return arr
    if len(arr) and arr.max(initial=-1) < offset:
        arr += offset
    return arr


def _npz_key(z: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> str | None:
    """Return the first present key from a backward-compatible alias list."""
    for key in candidates:
        if key in z.files:
            return key
    return None


def _positive_key(z: np.lib.npyio.NpzFile, split: str) -> str:
    key = _npz_key(
        z,
        (
            f"{split}_pos",
            f"{split}_positive",
            f"{split}_positives",
            f"{split}_edges",
            split,
        ),
    )
    if key is None:
        raise KeyError(
            f"Shared split archive is missing the positive array for {split!r}. "
            f"Available keys: {sorted(z.files)}"
        )
    return key


def _negative_key(z: np.lib.npyio.NpzFile, split: str) -> str | None:
    return _npz_key(
        z,
        (
            f"{split}_neg",
            f"{split}_negative",
            f"{split}_negatives",
            f"{split}_neg_tails",
            f"{split}_negative_tails",
            f"{split}_neg_edges",
            f"{split}_negative_edges",
        ),
    )


def _candidate_nodes(task: str, info: dict) -> np.ndarray:
    if task == "movie_link":
        return np.arange(info["off_link"], info["off_link"] + info["num_links"], dtype=np.int64)
    if task == "movie_year":
        return np.arange(info["off_year"], info["off_year"] + info["num_years"], dtype=np.int64)
    return np.arange(info["off_director"], info["off_actor"], dtype=np.int64)


def _validate_shared_positives(
    positives: dict[str, np.ndarray],
    task: str,
    info: dict,
    source: str,
) -> None:
    """Fail early when shared ids do not match this preprocessed IMDb frame."""
    combined = np.concatenate([positives[s] for s in ("train", "val", "test")], axis=0)
    if combined.ndim != 2 or combined.shape[1] != 2:
        raise ValueError(f"Invalid positive-pair shape in {source}: {combined.shape}")
    if len(combined) == 0:
        raise ValueError(f"Shared split archive contains no positives: {source}")

    heads = combined[:, 0]
    if heads.min(initial=0) < 0 or heads.max(initial=-1) >= info["num_movies"]:
        raise ValueError(
            f"Shared movie ids are outside [0,{info['num_movies']}) in {source}; "
            "the CSV filtering/order does not match the shared-split builder."
        )
    unique_heads, counts = np.unique(heads, return_counts=True)
    if np.any(counts != 1):
        raise ValueError(f"Shared splits repeat a movie head in {source}")
    if task == "movie_year":
        # 3 of 4180 movies have no title_year and therefore no target pair, so
        # Movie-Year covers a SUBSET of the movie set rather than partitioning it.
        # Everything else about the partition is still checked below.
        if len(unique_heads) > info["num_movies"]:
            raise ValueError(
                f"Movie-Year shared splits cover {len(unique_heads)} heads for "
                f"{info['num_movies']} movies in {source}."
            )
    else:
        expected_heads = np.arange(info["num_movies"], dtype=np.int64)
        if not np.array_equal(unique_heads, expected_heads):
            raise ValueError(
                f"Shared splits must partition every filtered movie exactly once. "
                f"Found {len(unique_heads)} unique heads for {info['num_movies']} movies in {source}."
            )

    # The builders partition by movie, so a movie may not occur in two splits.
    split_heads = {s: set(map(int, positives[s][:, 0])) for s in positives}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_heads[left] & split_heads[right]
        if overlap:
            raise ValueError(f"Shared split head overlap {left}/{right}: {len(overlap)} movies")

    tails = combined[:, 1]
    if task == "movie_link":
        lo, hi = info["off_link"], info["num_nodes"]
        if tails.min(initial=lo) < lo or tails.max(initial=lo - 1) >= hi:
            raise ValueError(f"Movie-Link tails outside [{lo},{hi}) in {source}")
        expected_tails = heads + info["off_link"]
        if not np.array_equal(tails, expected_tails):
            raise ValueError(
                "Movie-Link shared positives do not follow the expected one-to-one "
                "movie/link indexing. Regenerate the split with build_IMDB_ml_shared_splits.py."
            )
    elif task == "movie_year":
        lo, hi = info["off_year"], info["off_year"] + info["num_years"]
        if tails.min(initial=lo) < lo or tails.max(initial=lo - 1) >= hi:
            raise ValueError(f"Movie-Year tails outside [{lo},{hi}) in {source}")
    else:
        lo, hi = info["off_director"], info["off_actor"]
        if tails.min(initial=lo) < lo or tails.max(initial=lo - 1) >= hi:
            raise ValueError(f"Movie-Director tails outside [{lo},{hi}) in {source}")


def _sample_query_negatives(
    query_pos: np.ndarray,
    all_pos: np.ndarray,
    candidate_nodes: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Generate fixed, filtered tail negatives for one positive-query split."""
    if k <= 0:
        raise ValueError(f"neg_k must be positive, got {k}")
    rng = np.random.default_rng(seed)
    true_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in np.asarray(all_pos, dtype=np.int64):
        true_by_head[int(head)].add(int(tail))

    candidates = np.asarray(candidate_nodes, dtype=np.int64)
    out = np.empty((len(query_pos), k), dtype=np.int64)
    for row, (head, _tail) in enumerate(np.asarray(query_pos, dtype=np.int64)):
        forbidden = true_by_head[int(head)]
        valid = candidates[~np.isin(candidates, np.fromiter(forbidden, dtype=np.int64))]
        if len(valid) == 0:
            raise RuntimeError(f"No valid negative tails for head {head}")
        out[row] = rng.choice(valid, size=k, replace=len(valid) < k)
    return out


def _validate_negatives(
    negatives: np.ndarray,
    query_pos: np.ndarray,
    all_pos: np.ndarray,
    candidate_nodes: np.ndarray,
    split: str,
) -> None:
    arr = np.asarray(negatives, dtype=np.int64)
    query_pos = np.asarray(query_pos, dtype=np.int64)
    candidates = set(map(int, np.asarray(candidate_nodes, dtype=np.int64)))
    true_by_head: dict[int, set[int]] = defaultdict(set)
    for head, tail in np.asarray(all_pos, dtype=np.int64):
        true_by_head[int(head)].add(int(tail))

    is_tail_matrix = (
        arr.ndim == 2
        and arr.shape[0] == query_pos.shape[0]
        and (arr.shape[1] != 2 or not np.array_equal(arr[:, 0], query_pos[:, 0]))
    )
    if is_tail_matrix:
        for (head, _), row in zip(query_pos, arr):
            for tail in row:
                if int(tail) not in candidates:
                    raise ValueError(f"{split} negative tail {tail} is outside the target type")
                if int(tail) in true_by_head[int(head)]:
                    raise ValueError(f"{split} negatives contain true pair ({head},{tail})")
        return

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Unsupported {split} negative shape: {arr.shape}")
    for head, tail in arr:
        if int(tail) not in candidates:
            raise ValueError(f"{split} negative tail {tail} is outside the target type")
        if int(tail) in true_by_head[int(head)]:
            raise ValueError(f"{split} negatives contain true pair ({head},{tail})")


def load_or_create_lp_splits(
    shared_npz: str,
    blocks: list[dict],
    task: str,
    info: dict,
    neg_k: int,
    seed: int,
    my_neg_policy: str = "freq_matched",
) -> dict[str, torch.Tensor]:
    """Match ``preprocess_IMDB_rgcn_lp_skip.py`` exactly.

    The official skip preprocessor reads only ``train_pos/val_pos/test_pos`` from
    the shared archive and deterministically regenerates every negative matrix.
    We intentionally ignore any negative arrays that happen to be stored in the
    archive, because consuming them would define a different task.
    """
    if not shared_npz or not Path(shared_npz).is_file():
        raise FileNotFoundError(
            f"IMDb LP skip alignment requires the shared positive archive; missing: {shared_npz!r}"
        )
    with np.load(shared_npz) as z:
        missing = [k for k in ("train_pos", "val_pos", "test_pos") if k not in z.files]
        if missing:
            raise KeyError(f"Shared split archive missing exact skip keys {missing}: {shared_npz}")
        raw_pos = {split: np.asarray(z[f"{split}_pos"], dtype=np.int64) for split in ("train", "val", "test")}

    positive = {split: _convert_shared_pairs(raw_pos[split], task, info) for split in raw_pos}
    _validate_shared_positives(positive, task, info, shared_npz)

    k = int(neg_k)
    if task == "movie_link":
        k = min(k, max(1, info["num_movies"] - 1))
        all_true_local = set()
        for split in ("train", "val", "test"):
            for head, tail_global in positive[split]:
                all_true_local.add((int(head), int(tail_global - info["off_link"])))

        def sample_ml(pos_global: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
            out = np.zeros((len(pos_global), k), dtype=np.int64)
            all_links = np.arange(info["num_movies"], dtype=np.int64)
            for i, (head, _tail_global) in enumerate(pos_global):
                cand = [x for x in all_links if (int(head), int(x)) not in all_true_local]
                if not cand:
                    cand = list(all_links)
                out[i] = rng.choice(cand, size=k, replace=(k > len(cand))) + info["off_link"]
            return out

        train_neg = sample_ml(positive["train"], np.random.RandomState(seed))
        val_rng = np.random.RandomState(seed + 7)
        val_neg = sample_ml(positive["val"], val_rng)
        # Official code reuses the same RandomState object for val then test.
        test_neg = sample_ml(positive["test"], val_rng)
    elif task == "movie_year":
        # Year is a heavily SKEWED label (91 values, top year 216 movies), so the
        # negative-sampling policy decides what the metric measures. Measured on the
        # test split with a training-only oracle:
        #
        #   policy         random   frequency-prior alone   director+actor signal
        #   uniform         .1799            .4352                  .4457
        #   freq_matched    .1799            .1844                  .2379
        #
        # ``freq_matched`` draws negatives from the empirical year marginal, which
        # puts a pure label-frequency prior at chance and leaves only graph signal
        # above it. ``uniform`` is kept for comparability with the md/ml protocol.
        if my_neg_policy not in {"freq_matched", "uniform"}:
            raise ValueError(f"my_neg_policy must be freq_matched|uniform, got {my_neg_policy!r}")
        ny = info["num_years"]
        k = min(k, max(1, ny - 1))
        train_year_local = np.asarray(
            [int(t) - info["off_year"] for _h, t in positive["train"]], dtype=np.int64
        )
        marginal = np.bincount(train_year_local, minlength=ny).astype(np.float64)
        marginal /= marginal.sum()

        def sample_my(pos_global: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
            out = np.zeros((len(pos_global), k), dtype=np.int64)
            all_years = np.arange(ny, dtype=np.int64)
            for i, (_head, tail_global) in enumerate(pos_global):
                true_local = int(tail_global - info["off_year"])
                if my_neg_policy == "uniform":
                    cand = all_years[all_years != true_local]
                    picked = rng.choice(cand, size=k, replace=(k > len(cand)))
                else:
                    picked = np.empty(k, dtype=np.int64)
                    filled = 0
                    while filled < k:
                        draw = rng.choice(ny, size=4 * k, p=marginal)
                        draw = draw[draw != true_local]
                        take = min(k - filled, len(draw))
                        picked[filled:filled + take] = draw[:take]
                        filled += take
                out[i] = picked + info["off_year"]
            return out

        train_neg = sample_my(positive["train"], np.random.RandomState(seed))
        val_rng = np.random.RandomState(seed + 7)
        val_neg = sample_my(positive["val"], val_rng)
        test_neg = sample_my(positive["test"], val_rng)
    else:
        def sample_md(pos_global: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
            out = np.zeros((len(pos_global), k), dtype=np.int64)
            for i, (_head, tail_global) in enumerate(pos_global):
                true_local = int(tail_global - info["off_director"])
                cand = [d for d in range(info["num_directors"]) if d != true_local]
                if not cand:
                    cand = list(range(info["num_directors"]))
                out[i] = rng.choice(cand, size=k, replace=(k > len(cand))) + info["off_director"]
            return out

        train_neg = sample_md(positive["train"], np.random.RandomState(seed))
        val_rng = np.random.RandomState(seed + 7)
        val_neg = sample_md(positive["val"], val_rng)
        test_neg = sample_md(positive["test"], val_rng)

    task_data = {
        "train_pos": torch.tensor(positive["train"], dtype=torch.long),
        "val_pos": torch.tensor(positive["val"], dtype=torch.long),
        "test_pos": torch.tensor(positive["test"], dtype=torch.long),
        "train_neg": torch.tensor(train_neg, dtype=torch.long),
        "val_neg": torch.tensor(val_neg, dtype=torch.long),
        "test_neg": torch.tensor(test_neg, dtype=torch.long),
    }
    all_pos = np.concatenate([positive[s] for s in ("train", "val", "test")], axis=0)
    candidates = _candidate_nodes(task, info)
    for split in ("train", "val", "test"):
        _validate_negatives(
            task_data[f"{split}_neg"].numpy(), positive[split], all_pos, candidates, split
        )
    return task_data

def add_edge(store: dict[str, set[tuple[int, int]]], relation: str, source: int, target: int) -> None:
    store[relation].add((int(source), int(target)))


def build_physical(
    blocks: list[dict],
    variant: str,
    task: str,
    train_target_pairs: set[tuple[int, int]],
    ml_context: str = "all_movies",
) -> dict[str, set[tuple[int, int]]]:
    """Build physically distinct variants whose compiled LP graph equals skip.

    Movie-Director keeps Link->Director context for every movie, while
    Movie->Director is present only for training targets.

    Movie-Link has two selectable context policies:

    ``train_only`` reproduces the official skip contract
    (``preprocess_IMDB_rgcn_lp_skip.py``: ``if i not in train_movies: continue``),
    which drops non-training movie blocks ENTIRELY -- no actor, director or link
    edges. That leaves every val/test movie and link node isolated (measured:
    3254/12179 nodes with degree 0, and 418/418 val + 836/836 test endpoints at
    degree 0). With learned node embeddings and no input features those nodes get
    no gradient and no neighbourhood, so the model provably cannot rank them; all
    seeds collapse to a single constant score. Kept only for reproducing the old
    contract.

    ``all_movies`` (default) matches the ORIGINAL baseline
    (``preprocess_IMDB_rgcn_lp.build_ml``), which emits director/actor/link context
    for EVERY movie and restricts only the supervised Movie-Link target edges to
    the training split. That is the contract the original numbers we compare
    against were produced under, and it is leakage-free: the supervised pairs stay
    train-only; actor/director context for a held-out movie is not its label.

    ``all_pairs`` is DELIBERATELY LEAKY and exists only as a diagnostic. It emits
    the Movie-Link edge for every movie, so all 418 val and 836 test positives sit
    in the message graph verbatim. Movie-Link is the identity map (link i is movie
    i's IMDb URL, 4180 <-> 4180), so that edge IS the label. Two things follow, and
    they are the point of keeping the mode:
      * the task becomes trivially solvable, reproducing the draft's Movie-Link
        numbers; and
      * cross-variant invariance is RESTORED, because ``compile_semantic`` walks
        movie_to_link -> link_to_{director,actor} to project link-side context onto
        movie-side semantic edges. With the target edge train-only, held-out movies
        have no bridge, so v1 (attributes on the movie) and v2 (attributes on the
        link) compile to different semantic graphs -- four distinct semantic hashes.
        The edge the compiler needs as CONTEXT and the edge being PREDICTED are the
        same edge, which is why this task cannot be both honest and invariant.
    Never report ``all_pairs`` output as performance.
    """
    if ml_context not in {"all_movies", "train_only", "all_pairs"}:
        raise ValueError(
            f"ml_context must be all_movies|train_only|all_pairs, got {ml_context!r}"
        )
    if variant not in PLACEMENT[task]:
        raise ValueError(f"Unknown variant {variant!r} for task {task!r}")
    place = PLACEMENT[task][variant]
    raw: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for block in blocks:
        movie, director, link = block["movie"], block["director"], block["link"]
        year = block.get("year")
        target_node = {"movie_link": link, "movie_director": director,
                       "movie_year": year}.get(task)
        target_pair = (movie, target_node)
        target_is_train = task == "nc" or target_pair in train_target_pairs

        if task == "movie_link" and ml_context == "train_only" and not target_is_train:
            continue

        # The supervised Movie-Link edge is training-only under ``train_only`` and
        # ``all_movies``; context edges below are emitted for every movie under
        # ``all_movies``. ``all_pairs`` DELIBERATELY LEAKS: it emits the supervised
        # Movie-Link edge for every block, val/test included. See the docstring.
        if task != "movie_link" or target_is_train or ml_context == "all_pairs":
            add_edge(raw, "movie_to_link", movie, link)
            add_edge(raw, "link_to_movie", link, movie)

        if task == "movie_director":
            # link_to_director / director_to_link are NOT emitted for movie_director.
            #
            # History: they were originally emitted for EVERY movie as "non-target
            # skip context". That leaked -- Link is a BIJECTION with Movie, so
            # link_to_director(link(m), d) IS the supervised movie_to_director(m, d)
            # under another node type, and with movie_to_link present for every
            # movie each held-out movie had a 2-hop path to its own answer
            # (audited: 1254/1254 val+test labels reachable). Guarding them to
            # training pairs closed the leak but left a train/test mismatch: a
            # TRAIN positive was then reachable by TWO routes and a TEST positive
            # by NONE, giving the model a shortcut to lean on that vanishes at
            # inference. Dropping the relation removes both problems.
            #
            # Invariance is unaffected either way: the attribute RELOCATED between
            # MD's variants is the ACTOR, not the director, so v1 and v3 still
            # compile to a single semantic program.
            if target_is_train:
                add_edge(raw, "movie_to_director", movie, director)
                add_edge(raw, "director_to_movie", director, movie)
        else:
            for endpoint in place["director"]:
                head = movie if endpoint == "movie" else link
                add_edge(raw, f"{endpoint}_to_director", head, director)
                add_edge(raw, f"director_to_{endpoint}", director, head)

        actors = [block["actor1"]] if task == "movie_link" else block["actors"]
        for actor in actors:
            for endpoint in place["actor"]:
                head = movie if endpoint == "movie" else link
                add_edge(raw, f"{endpoint}_to_actor", head, actor)
                add_edge(raw, f"actor_to_{endpoint}", actor, head)

        if year is not None:
            if task == "movie_year":
                # The target relation: emitted for TRAINING pairs only, and never
                # placed on the link, exactly as movie_to_director is for MD.
                if target_is_train:
                    add_edge(raw, "movie_to_year", movie, year)
                    add_edge(raw, "year_to_movie", year, movie)
            else:
                for endpoint in place.get("year", ()):
                    head = movie if endpoint == "movie" else link
                    add_edge(raw, f"{endpoint}_to_year", head, year)
                    add_edge(raw, f"year_to_{endpoint}", year, head)
    return raw

def _adj(edges: dict[str, set[tuple[int, int]]], relation: str) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for source, target in edges.get(relation, set()):
        out[source].add(target)
    return out


def compile_semantic(raw: dict[str, set[tuple[int, int]]], task: str) -> tuple[dict[str, set[tuple[int, int]]], dict]:
    """Compile distinct physical contexts into one transformation-closed RGCN bias.

    Direct edges are propagation edges. In reified contexts, the Link node is
    context-only: M-L-X emits the semantic M-X message. The inverse reconstruction
    L-M-X emits L-X. Set semantics removes representation-dependent witness counts.
    """
    semantic: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for relation, edges in raw.items():
        semantic[relation].update(edges)

    m_to_l = _adj(raw, "movie_to_link")
    l_to_m = _adj(raw, "link_to_movie")
    m_to_d = _adj(raw, "movie_to_director")
    l_to_d = _adj(raw, "link_to_director")
    m_to_a = _adj(raw, "movie_to_actor")
    l_to_a = _adj(raw, "link_to_actor")
    m_to_y = _adj(raw, "movie_to_year")
    l_to_y = _adj(raw, "link_to_year")
    audit = defaultdict(int)

    for movie, links in m_to_l.items():
        for link in links:
            for director in l_to_d.get(link, ()):
                # For Movie-Director LP this is the held-out target relation.  The
                # official skip graph keeps Link->Director as context but never
                # reconstructs held-out Movie->Director message edges.
                if task != "movie_director":
                    add_edge(semantic, "movie_to_director", movie, director)
                    add_edge(semantic, "director_to_movie", director, movie)
                    audit["MLD_to_MD"] += 1
            for actor in l_to_a.get(link, ()):
                add_edge(semantic, "movie_to_actor", movie, actor)
                add_edge(semantic, "actor_to_movie", actor, movie)
                audit["MLA_to_MA"] += 1
            for year in l_to_y.get(link, ()):
                # Reachable only if some variant puts the year on the link, which none
                # currently does; kept so the projection stays symmetric with the actor
                # and director rules if a future family relocates the year.
                if task != "movie_year":
                    add_edge(semantic, "movie_to_year", movie, year)
                    add_edge(semantic, "year_to_movie", year, movie)
                    audit["MLY_to_MY"] += 1

    for link, movies in l_to_m.items():
        for movie in movies:
            for director in m_to_d.get(movie, ()):
                add_edge(semantic, "link_to_director", link, director)
                add_edge(semantic, "director_to_link", director, link)
                audit["LMD_to_LD"] += 1
            for actor in m_to_a.get(movie, ()):
                add_edge(semantic, "link_to_actor", link, actor)
                add_edge(semantic, "actor_to_link", actor, link)
                audit["LMA_to_LA"] += 1
            for year in m_to_y.get(movie, ()):
                # Project the year onto the link, exactly as the director and actor
                # rules do. movie_to_year and movie_to_link are both present in EVERY
                # variant, so link -> movie -> year is derivable identically in every
                # variant: this is recoverable information, not the compiler reaching
                # outside the family.
                #
                # SUPPRESSED for movie_year, where movie_to_year is the held-out TARGET
                # and carries training pairs only. Synthesising from it would give the
                # 2925 training movies' links a year edge and the 834 test movies' links
                # none -- the train/test structure gap diagnosed on Movie-Director's
                # link_to_director. The general rule is:
                # project across the bridge always, EXCEPT from a held-out target
                # relation -- the same rule that already stops movie_to_director and
                # movie_to_year being reconstructed.
                if task != "movie_year":
                    add_edge(semantic, "link_to_year", link, year)
                    add_edge(semantic, "year_to_link", year, link)
                    audit["LMY_to_LY"] += 1
    return semantic, dict(audit)


def official_skip_semantic_multiplicity(
    blocks: list[dict], task: str, train_target_pairs: set[tuple[int, int]],
    ml_context: str = "all_movies",
) -> dict[str, dict[tuple[int, int], int]]:
    """Parallel-edge multiplicities of the official skip universal graph.

    The custom tensor encoder collapses identical pairs but multiplies their
    messages by this weight, which is algebraically equivalent to DGL parallel
    edges under the skip runner's unnormalized sum aggregation.
    """
    counts: dict[str, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    def add(name: str, u: int, v: int, n: int = 1):
        counts[name][(int(u), int(v))] += int(n)
    for block in blocks:
        movie, director, link = block["movie"], block["director"], block["link"]
        target = (movie, block.get(TARGET_KEY[task]))
        target_is_train = target in train_target_pairs
        if task == "movie_link" and ml_context == "train_only" and not target_is_train:
            continue
        # supervised Movie-Link edges stay training-only unless ml_context leaks them
        if task != "movie_link" or target_is_train or ml_context == "all_pairs":
            add("movie_to_link", movie, link); add("link_to_movie", link, movie)
        add("link_to_director", link, director); add("director_to_link", director, link)
        if task == "movie_link" or target_is_train:
            add("movie_to_director", movie, director); add("director_to_movie", director, movie)
        actors = [block["actor1"]] if task == "movie_link" else block["actors"]
        for actor in actors:
            add("movie_to_actor", movie, actor); add("actor_to_movie", actor, movie)
            add("link_to_actor", link, actor); add("actor_to_link", actor, link)
    return {name: dict(rows) for name, rows in counts.items()}


def to_tensors(
    edges: dict[str, set[tuple[int, int]]],
    multiplicity: dict[str, dict[tuple[int, int], int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target, relation_ids = [], [], []
    for relation in REL_NAMES:
        for u, v in sorted(edges.get(relation, set())):
            source.append(u)
            target.append(v)
            relation_ids.append(REL[relation])
    edge_index, edge_type, edge_weight = canonicalize_edges(
        torch.tensor([source, target], dtype=torch.long),
        torch.tensor(relation_ids, dtype=torch.long),
    )
    if multiplicity is not None and edge_type.numel():
        values = []
        for rid, u, v in zip(edge_type.tolist(), edge_index[0].tolist(), edge_index[1].tolist()):
            name = REL_NAMES[int(rid)]
            values.append(float(multiplicity.get(name, {}).get((int(u), int(v)), 1)))
        edge_weight = torch.tensor(values, dtype=torch.float32)
    return edge_index, edge_type, edge_weight


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build physically distinct IMDb variants with architecture-level conditional RGCN messages"
    )
    ap.add_argument("--raw-csv", default="data/raw/IMDB/movie_metadata.csv")
    ap.add_argument("--out-root", default="data/preprocessed/RGCN_structural")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--shared-movie-link", default="", help="Optional repository-aligned IMDB_ml_shared_splits.npz")
    ap.add_argument("--shared-movie-director", default="", help="Optional repository-aligned IMDB_md_shared_splits.npz")
    ap.add_argument("--shared-movie-year", default="", help="Repository-aligned IMDB_my_shared_splits.npz")
    ap.add_argument(
        "--my-neg-policy",
        choices=("freq_matched", "uniform"),
        default="freq_matched",
        help=(
            "Movie-Year negative sampling. 'freq_matched' (default) draws negatives "
            "from the empirical year marginal so a pure label-frequency prior scores "
            "at chance and only graph signal shows above it; 'uniform' draws years "
            "uniformly, which lets that prior carry most of the metric "
            "(measured .4352 of .4457)."
        ),
    )
    ap.add_argument("--feature-dim", type=int, default=128,
                    help="Only used by --feature-mode svd_movie_only.")
    ap.add_argument(
        "--feature-mode",
        choices=("pooled_bow", "svd_movie_only"),
        default="pooled_bow",
        help=(
            "NC node features. 'pooled_bow' (default) = the canonical table "
            "shared with SlotGAT / the RGCN conditional package / MAGNN: raw plot_keywords "
            "BoW on movies, mean-of-member-movies on every other type, ~3008-d. "
            "'svd_movie_only' = the legacy 128-d SVD movie features with zero rows on all "
            "other types (the .5965 run). LP tasks are unaffected: they carry no x and use "
            "learned node embeddings, as before."
        ),
    )
    ap.add_argument("--neg-k", type=int, default=19)
    ap.add_argument(
        "--ml-context",
        choices=("all_movies", "train_only", "all_pairs"),
        default="all_movies",
        help=(
            "Movie-Link message-graph context policy. 'all_movies' (default) emits "
            "director/actor/link context for every movie and keeps only the "
            "supervised Movie-Link edges training-only, matching the ORIGINAL "
            "baseline this task is compared against. 'train_only' reproduces the "
            "official skip contract, which isolates every val/test node and makes "
            "the task unlearnable -- kept only for reproducing old results. "
            "'all_pairs' DELIBERATELY LEAKS the label: it emits the Movie-Link edge "
            "for EVERY movie, so all 418 val and 836 test positives are in the "
            "message graph verbatim. Because Movie-Link is the identity, that edge "
            "IS the answer. It exists only to reproduce the draft's Movie-Link "
            "numbers and to demonstrate that the compiler's cross-variant bridge "
            "and the label are the same edge -- NEVER report it as performance."
        ),
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    set_deterministic(args.seed)
    variants, tasks = parse_str_list(args.variants), parse_str_list(args.tasks)
    if set(variants) - set(VARIANTS):
        raise SystemExit(f"Unknown variants: {set(variants) - set(VARIANTS)}")
    if set(tasks) - set(TASKS):
        raise SystemExit(f"Unknown tasks: {set(tasks) - set(TASKS)}")

    for task in tasks:
        movies, labels = load_movies(args.raw_csv, task)
        info = build_entities(movies, task)
        blocks = build_blocks(movies, info, task)
        node_type = torch.tensor(info["node_type"], dtype=torch.long)
        y = torch.full((info["num_nodes"],), -1, dtype=torch.long)
        y[:info["num_movies"]] = torch.tensor(labels, dtype=torch.long)

        if task == "nc":
            split_ids = split_movie_ids(info["num_movies"], args.seed)
            task_data: dict[str, torch.Tensor] = {}
            train_target_pairs: set[tuple[int, int]] = set()
        else:
            shared = {
                "movie_link": args.shared_movie_link,
                "movie_director": args.shared_movie_director,
                "movie_year": args.shared_movie_year,
            }[task]
            task_data = load_or_create_lp_splits(
                shared, blocks, task, info, args.neg_k, args.seed,
                my_neg_policy=args.my_neg_policy,
            )
            train_target_pairs = set(map(tuple, task_data["train_pos"].cpu().numpy().tolist()))
            split_ids = {"train": np.array([], dtype=np.int64), "val": np.array([], dtype=np.int64), "test": np.array([], dtype=np.int64)}

        # Features are built after the split is known: for LP tasks the held-out target
        # type pools over TRAINING pairs only (canonical rule). 'svd_movie_only'
        # keeps the legacy behaviour -- SVD movie features on nc, NO x (learned embeddings)
        # on LP.
        if args.feature_mode == "pooled_bow":
            features = build_features_canonical(movies, info, task, train_target_pairs)
            x_tensor = torch.tensor(features)
        else:
            features = build_features_svd_movie_only(movies, info, args.feature_dim, args.seed)
            x_tensor = torch.tensor(features) if task == "nc" else None
        feature_meta = {
            "feature_mode": args.feature_mode if x_tensor is not None else "none_learned_embeddings",
            "feature_rule": (CANONICAL_FEATURE_RULE + ("+target_type_train_only" if task in LP_TARGET_TYPE else "")) if args.feature_mode == "pooled_bow" else "svd_movie_only_zero_others",
            "feature_dim": int(features.shape[1]) if x_tensor is not None else None,
            "feature_table_hash": feature_table_hash(features) if x_tensor is not None else None,
            "lp_target_type_pooled_train_only": LP_TARGET_TYPE.get(task) if (x_tensor is not None and task in LP_TARGET_TYPE) else None,
        }
        print(f"[{task}] features: mode={feature_meta['feature_mode']} shape={tuple(features.shape) if x_tensor is not None else None} hash={(feature_meta['feature_table_hash'] or '')[:16]}")

        masks = {}
        for split, ids in split_ids.items():
            mask = torch.zeros(info["num_nodes"], dtype=torch.bool)
            if len(ids):
                mask[torch.tensor(ids, dtype=torch.long)] = True
            masks[split] = mask

        # Each task has its own variant family (PLACEMENT): movie_director relocates
        # only the actor and so has v1/v3, while nc/movie_link/movie_year relocate a
        # pair and have v1..v4. Asking for the whole default list is therefore normal;
        # variants outside a task's family are skipped rather than treated as an error.
        task_variants = [v for v in variants if v in PLACEMENT[task]]
        skipped = [v for v in variants if v not in PLACEMENT[task]]
        if skipped:
            print(f"[{task}] not in this task's variant family, skipping: {skipped}")
        if not task_variants:
            raise SystemExit(f"No requested variant is defined for task {task}")
        semantic_hashes, physical_hashes = [], []
        for variant in task_variants:
            raw = build_physical(blocks, variant, task, train_target_pairs,
                                 ml_context=args.ml_context)
            semantic, context_audit = compile_semantic(raw, task)
            physical_index, physical_type, _ = to_tensors(raw)
            # Parallel-edge multiplicities exist only to reproduce the OFFICIAL SKIP
            # universal graph, which is the baseline movie_link and movie_director are
            # compared against. Movie-Year has no such baseline, so it uses plain set
            # semantics (every semantic edge weight 1), matching the declared
            # "message_semantics": "set" metadata.
            multiplicity = (
                official_skip_semantic_multiplicity(blocks, task, train_target_pairs,
                                                    ml_context=args.ml_context)
                if task in ("movie_link", "movie_director") else None
            )
            semantic_index, semantic_type, semantic_weight = to_tensors(semantic, multiplicity)
            data = ConditionalGraphData(
                variant=variant,
                num_nodes=info["num_nodes"],
                node_type=node_type,
                physical_edge_index=physical_index,
                physical_edge_type=physical_type,
                physical_relation_names=REL_NAMES,
                semantic_edge_index=semantic_index,
                semantic_edge_type=semantic_type,
                semantic_edge_weight=semantic_weight,
                semantic_relation_names=REL_NAMES,
                x=x_tensor,
                y=y,
                train_mask=masks["train"],
                val_mask=masks["val"],
                test_mask=masks["test"],
                task_data=task_data,
                context_audit=context_audit,
                metadata={
                    "dataset": "IMDB",
                    "task": task,
                    "num_classes": 3,
                    "num_nodes_by_type": {
                        "movie": info["num_movies"],
                        "director": info["num_directors"],
                        "actor": info["num_actors"],
                        "link": info["num_links"],
                        "year": info["num_years"],
                    },
                    "offsets": {
                        name: int(info[f"off_{name}"])
                        for name in ("movie", "director", "actor", "link", "year")
                    },
                    "message_semantics": "set",
                    "target_edges_hidden": task != "nc",
                    "shared_split_file": shared if task != "nc" and shared else None,
                    "movie_link_leakage_policy": "official_skip_train_blocks_only" if task == "movie_link" else None,
                    "skip_task_contract": "exact_shared_positives_regenerated_negatives_k_matrix",
                    "skip_neg_k": int(args.neg_k),
                    "variant_placement": PLACEMENT[task][variant],
                    "movie_year_neg_policy": args.my_neg_policy if task == "movie_year" else None,
                    **feature_meta,
                },
            )
            out = Path(args.out_root) / "IMDB" / task / variant
            save_graph_data(data, out)
            semantic_hashes.append(data.semantic_hash)
            physical_hashes.append(data.physical_hash)
            print(
                f"[{task}/{variant}] physical={data.physical_hash[:12]} semantic={data.semantic_hash[:12]} "
                f"Eraw={physical_type.numel()} Esem={semantic_type.numel()}"
            )
        if len(set(semantic_hashes)) != 1:
            raise RuntimeError(f"Semantic program mismatch for task {task}: {semantic_hashes}")
        if len(set(physical_hashes)) < 2:
            raise RuntimeError(f"Physical graphs unexpectedly identical for task {task}")
    print("IMDb preprocessing complete: raw hashes differ; semantic hashes match within each task.")


if __name__ == "__main__":
    main()
