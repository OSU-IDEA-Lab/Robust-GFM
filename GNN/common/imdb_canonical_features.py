"""Canonical IMDb node features, shared by every baseline.

One definition, importable from any baseline directory, so that RGCN, SlotGAT, MAGNN and
SeHGNN feed the SAME node inputs on the IMDb tasks.

The rule
--------
* movie  : bag of words over ``plot_keywords`` ("|" -> " "), ``CountVectorizer(min_df=2)``
           fit on ALL movies of the task's frame (transductive vocabulary, ~3008 terms),
           RAW counts, no row normalisation.  This is the paper's MAGNN convention.
* every other type (director, actor, link, year): the MEAN of the BoW rows of the movies
           the node is attached to IN THE DATAFRAME -- director_name, actor_1/2/3_name,
           the movie itself for its link, title_year.  Membership comes from the frame,
           never from a variant's graph, so the table is identical in every variant by
           construction.  A node with no member movies gets a zero row.
* link prediction exception: for the task's HELD-OUT target type (director in
           movie_director, year in movie_year, link in movie_link) membership is
           restricted to TRAINING pairs, so a val/test movie contributes nothing to its
           true target's feature.  MAGNN's LP preprocessor already does this (it pools
           from an adjacency that carries only ``train_pos`` target edges).

Provenance of the rule: ``MAGNN/preprocess_IMDB_star.py`` (movie BoW + director/actor
means), ``RGCN/preprocess_IMDB_rgcn_conditional.py:build_all_node_features`` (the same
mean rule extended to the link), ``SlotGAT/run_IMDB_nc_conditional.py:
build_pooled_bow_features`` (verified byte-identical to the RGCN conditional table).
The one known deviation is MAGNN's LINK row (mean of the movie's director+actor rows,
a two-hop mean) -- to be replaced by this rule when MAGNN is rebuilt on the year graph.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

CANONICAL_FEATURE_RULE = "imdb_bow_min_df2_raw_counts+mean_of_member_movies_v1"
ACTOR_COLUMNS = ("actor_1_name", "actor_2_name", "actor_3_name")


def movie_bow(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Raw-count BoW of plot_keywords, ``CountVectorizer(min_df=2)``, float32 dense."""
    text = movies["plot_keywords"].fillna("").map(lambda v: str(v).replace("|", " "))
    vectorizer = CountVectorizer(min_df=2)
    sparse = vectorizer.fit_transform(text)
    return sparse.toarray().astype(np.float32), list(vectorizer.get_feature_names_out())


def pooled_mean(movie_x: np.ndarray, groups: Sequence[Iterable[int]]) -> np.ndarray:
    """Row i = mean of ``movie_x`` over ``groups[i]``; an empty group gives a zero row."""
    out = np.zeros((len(groups), movie_x.shape[1]), dtype=np.float32)
    for i, members in enumerate(groups):
        ids = np.fromiter((int(m) for m in members), dtype=np.int64)
        if ids.size:
            out[i] = movie_x[ids].mean(axis=0)
    return out


def frame_membership(
    movies: pd.DataFrame,
    director_to_idx: Mapping[str, int],
    actor_to_idx: Mapping[str, int],
    year_to_idx: Mapping[int, int] | None = None,
    actor_columns: Sequence[str] = ACTOR_COLUMNS,
    train_only: Mapping[str, Iterable[tuple[int, int]]] | None = None,
) -> dict[str, list[list[int]]]:
    """Member-movie lists per node, in node order, straight from the dataframe.

    ``train_only`` maps a target type name ("director", "year", "link") to the
    (movie_index, local_target_index) TRAINING pairs; that type's membership is then
    built from those pairs only (the LP exception above).
    """
    n_movies = len(movies)
    members: dict[str, list[list[int]]] = {
        "director": [[] for _ in range(len(director_to_idx))],
        "actor": [[] for _ in range(len(actor_to_idx))],
        "link": [[i] for i in range(n_movies)],
    }
    if year_to_idx is not None:
        members["year"] = [[] for _ in range(len(year_to_idx))]
    train_only = dict(train_only or {})

    for movie_idx, row in movies.iterrows():
        movie_idx = int(movie_idx)
        director = row.get("director_name")
        if "director" not in train_only and pd.notna(director) and director in director_to_idx:
            members["director"][director_to_idx[director]].append(movie_idx)
        for column in actor_columns:
            actor = row.get(column)
            if pd.notna(actor) and actor in actor_to_idx:
                members["actor"][actor_to_idx[actor]].append(movie_idx)
        if year_to_idx is not None and "year" not in train_only:
            raw_year = row.get("title_year")
            if pd.notna(raw_year) and int(raw_year) in year_to_idx:
                members["year"][year_to_idx[int(raw_year)]].append(movie_idx)

    for type_name, pairs in train_only.items():
        if type_name not in members:
            raise KeyError(f"train_only names unknown type {type_name!r}")
        members[type_name] = [[] for _ in members[type_name]]
        for movie_idx, local_target in pairs:
            members[type_name][int(local_target)].append(int(movie_idx))
    # de-duplicate while keeping order (an actor credited twice on one movie counts once)
    for type_name, groups in members.items():
        members[type_name] = [sorted(set(g)) for g in groups]
    return members


def build_canonical_features(
    movies: pd.DataFrame,
    members: Mapping[str, Sequence[Iterable[int]]],
    type_order: Sequence[str],
) -> np.ndarray:
    """Stack [movie BoW] + [pooled rows per type, in ``type_order``] into one table."""
    movie_x, _ = movie_bow(movies)
    parts = [movie_x]
    for type_name in type_order:
        parts.append(pooled_mean(movie_x, members[type_name]))
    return np.concatenate(parts, axis=0).astype(np.float32)


def feature_table_hash(x: np.ndarray) -> str:
    """sha256 over shape + float32 bytes; equal hashes mean bit-identical tables."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()
