"""
Preprocess raw IMDB data into 5 graph variants in HGB format for SlotGAT.

Node types: 0=Movie, 1=Director, 2=Actor, 3=Link (IMDB link node), 4=Year
Task:       Movie genre classification (Action/Comedy/Drama, 3 classes)

Variant connectivity (all share movie-link edges):
  1: actor-movie,  director-movie,  year-movie
  2: actor-link,   director-link,   year-movie
  3: actor-link,   director-movie,  year-movie
  4: actor-movie,  director-link,   year-movie
  5: actor+director on both endpoints, year-movie (the union of 1-4, not a variant)

The YEAR node is FIXED CONTEXT on the
movie in every transformation variant -- NC already has two relocatable attributes
({director, actor}), so year enters as extra context rather than as a third degree of
freedom, which would give 2^3 = 8 variants and break the IMDB_var1..5 naming every NC
runner depends on. Variant 5 is the union of variants 1-4, so it carries
{director, actor} on both endpoints and the year on the movie, where every variant has it.

Year nodes are appended AFTER the link block, so every pre-existing node id and
relation id is unchanged.

Output per variant (HGB format):
  node.dat, link.dat, label.dat, label.dat.test, info.dat

Usage:
    python preprocess_IMDB.py
"""

import os
import json
import pathlib
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = os.environ.get('INV_GNN_DATA', os.path.join(SCRIPT_DIR, 'data'))
RAW_CSV = os.path.join(SAVE_ROOT, 'raw', 'IMDB', 'movie_metadata.csv')
RAND_SEED = 1566911444

VARIANT_EDGES = {
    1: {'actor': 'movie',  'director': 'movie', 'year': 'movie'},
    2: {'actor': 'link',   'director': 'link',  'year': 'movie'},
    3: {'actor': 'link',   'director': 'movie', 'year': 'movie'},
    4: {'actor': 'movie',  'director': 'link',  'year': 'movie'},
    # Variant 5 is the UNION of variants 1-4, not a transformation variant: {director,
    # actor} relocate so the union carries them on both endpoints, while the year is
    # fixed on the movie in every variant and so stays on the movie here too.
    5: {'actor': ('movie', 'link'), 'director': ('movie', 'link'), 'year': 'movie'},
}

# Attribute roles in the order their relation ids are allocated. Year is LAST so the
# relation ids of every pre-existing relation are unchanged.
ROLES = ('director', 'actor', 'year')

# Node type ids. Year is appended last for the same reason.
NTYPE = {'movie': 0, 'director': 1, 'actor': 2, 'link': 3, 'year': 4}


def _normalize_targets(target_spec):
    if isinstance(target_spec, str):
        targets = [target_spec]
    else:
        targets = list(target_spec)

    seen = set()
    normalized = []
    for t in targets:
        if t not in {'movie', 'link'}:
            raise ValueError(f'Unsupported target type: {t}')
        if t not in seen:
            seen.add(t)
            normalized.append(t)
    return normalized


def load_and_clean():
    movies = (
        pd.read_csv(RAW_CSV, encoding='utf-8')
        .drop_duplicates(subset=['movie_imdb_link'])
        .dropna(axis=0, subset=['actor_1_name', 'director_name'])
        .reset_index(drop=True)
    )

    labels = np.zeros(len(movies), dtype=int)
    for idx, genres in movies['genres'].items():
        labels[idx] = -1
        for genre in genres.split('|'):
            if genre == 'Action':
                labels[idx] = 0
                break
            elif genre == 'Comedy':
                labels[idx] = 1
                break
            elif genre == 'Drama':
                labels[idx] = 2
                break

    unwanted = np.where(labels == -1)[0]
    movies = movies.drop(unwanted).reset_index(drop=True)
    labels = np.delete(labels, unwanted, 0)

    directors = sorted(set(movies['director_name'].dropna()))
    actors = sorted(set(
        movies['actor_1_name'].dropna().to_list() +
        movies['actor_2_name'].dropna().to_list() +
        movies['actor_3_name'].dropna().to_list()
    ))
    # Stable Link-node IDs across clean preprocessing runs.
    imdb_links = sorted(set(movies['movie_imdb_link'].dropna()))
    # 3 of 4180 movies have no title_year; they keep every other edge and simply
    # carry no year edge. 91 distinct values, 45.9 movies/year.
    years = sorted({int(y) for y in movies['title_year'].dropna()})

    return movies, labels, directors, actors, imdb_links, years


def build_edges(movies, directors, actors, imdb_links, years, variant_cfg):
    """Build list of (src_id, dst_id, relation_id, weight) tuples for a variant."""
    n_m, n_d, n_a, n_l = len(movies), len(directors), len(actors), len(imdb_links)
    off = {'movie': 0, 'director': n_m, 'actor': n_m + n_d,
           'link': n_m + n_d + n_a, 'year': n_m + n_d + n_a + n_l}

    targets = {role: _normalize_targets(variant_cfg[role])
               for role in ROLES if role in variant_cfg}

    edges = []
    rel_meta = {}

    # movie-link edges (always present): rel 0 = movie->link, rel 1 = link->movie
    rel_meta[0] = (NTYPE['movie'], NTYPE['link'])
    rel_meta[1] = (NTYPE['link'], NTYPE['movie'])

    rel_lookup = {}
    next_rel = 2
    for role in ROLES:
        for target in targets.get(role, ()):
            rel_meta[next_rel] = (NTYPE[target], NTYPE[role])
            rel_meta[next_rel + 1] = (NTYPE[role], NTYPE[target])
            rel_lookup[(role, target, 'forward')] = next_rel
            rel_lookup[(role, target, 'backward')] = next_rel + 1
            next_rel += 2

    director_idx = {name: i for i, name in enumerate(directors)}
    actor_idx = {name: i for i, name in enumerate(actors)}
    link_idx = {name: i for i, name in enumerate(imdb_links)}
    year_idx = {y: i for i, y in enumerate(years)}

    def emit(role, head_global, tail_global):
        for target in targets.get(role, ()):
            src = head_global[target]
            edges.append((src, tail_global, rel_lookup[(role, target, 'forward')], 1.0))
            edges.append((tail_global, src, rel_lookup[(role, target, 'backward')], 1.0))

    for movie_idx, row in movies.iterrows():
        link_global = off['link'] + link_idx[row['movie_imdb_link']]
        head = {'movie': movie_idx, 'link': link_global}

        # movie-link
        edges.append((movie_idx, link_global, 0, 1.0))
        edges.append((link_global, movie_idx, 1, 1.0))

        if row['director_name'] in director_idx:
            emit('director', head, off['director'] + director_idx[row['director_name']])

        for acol in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
            aname = row.get(acol)
            if pd.notna(aname) and aname in actor_idx:
                emit('actor', head, off['actor'] + actor_idx[aname])

        raw_year = row.get('title_year')
        if pd.notna(raw_year) and int(raw_year) in year_idx:
            emit('year', head, off['year'] + year_idx[int(raw_year)])

    return edges, rel_meta


def write_node_dat(path, movies, directors, actors, imdb_links, years, movie_features):
    """Write node.dat in HGB format."""
    n_m = len(movies)
    n_d = len(directors)
    n_a = len(actors)
    off_d = n_m
    off_a = n_m + n_d
    off_l = n_m + n_d + n_a
    off_y = off_l + len(imdb_links)

    with open(path, 'w', encoding='utf-8') as f:
        for i in range(n_m):
            name = str(movies.iloc[i]['movie_title']).strip().replace('\t', '_').replace(' ', '_')
            feat_str = ','.join(f'{v:.6f}' for v in movie_features[i])
            f.write(f'{i}\t{name}\t0\t{feat_str}\n')
        for i, dname in enumerate(directors):
            nid = off_d + i
            name = dname.strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{nid}\t{name}\t1\n')
        for i, aname in enumerate(actors):
            nid = off_a + i
            name = aname.strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{nid}\t{name}\t2\n')
        for i, link in enumerate(imdb_links):
            nid = off_l + i
            name = link.strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{nid}\t{name}\t3\n')
        for i, year in enumerate(years):
            nid = off_y + i
            f.write(f'{nid}\t{int(year)}\t4\n')


def write_link_dat(path, edges):
    """Write link.dat in HGB format."""
    with open(path, 'w', encoding='utf-8') as f:
        for (src, dst, rid, w) in edges:
            f.write(f'{src}\t{dst}\t{rid}\t{w}\n')


def write_label_dat(path, movie_indices, movies, labels):
    """Write label.dat or label.dat.test."""
    with open(path, 'w', encoding='utf-8') as f:
        for idx in movie_indices:
            name = str(movies.iloc[idx]['movie_title']).strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{idx}\t{name}\t0\t{labels[idx]}\n')


def write_info_dat(path, movies, directors, actors, imdb_links, years, rel_meta, feat_dim):
    """Write info.dat with JSON metadata."""
    info = {
        "node.dat": {
            "node type": {
                "0": "movie",
                "1": "director",
                "2": "actor",
                "3": "link",
                "4": "year"
            },
            "num_nodes": {
                "0": len(movies),
                "1": len(directors),
                "2": len(actors),
                "3": len(imdb_links),
                "4": len(years)
            },
            "feature_dim": feat_dim
        },
        "link.dat": {
            "link type": {}
        },
        "label.dat": {
            "num_classes": 3,
            "label type": {
                "0": "Action",
                "1": "Comedy",
                "2": "Drama"
            }
        }
    }
    type_names = {v: k for k, v in NTYPE.items()}
    for rid, (h, t) in sorted(rel_meta.items()):
        info["link.dat"]["link type"][str(rid)] = f"{type_names[h]}-{type_names[t]}"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--save-root', default=SAVE_ROOT)
    ap.add_argument('--suffix', default='',
                    help="Appended to each IMDB_var<N> directory name, so a rebuild "
                         "lands beside the existing one instead of overwriting it. "
                         "Point the runner at the same name via its variant map.")
    args = ap.parse_args()

    print('Loading raw IMDB data ...')
    movies, labels, directors, actors, imdb_links, years = load_and_clean()
    print(f'  Movies: {len(movies)}, Directors: {len(directors)}, '
          f'Actors: {len(actors)}, Links: {len(imdb_links)}, Years: {len(years)}')
    print(f'  Labels: {len(labels)}, classes: {np.unique(labels).tolist()}')

    # Movie BoW features from plot_keywords (shared across variants)
    vectorizer = CountVectorizer(min_df=2)
    movie_X = vectorizer.fit_transform(
        movies['plot_keywords'].fillna('').apply(
            lambda x: str(x).replace('|', ' ')
        ).values
    )
    movie_features = movie_X.toarray().astype(float)
    feat_dim = movie_features.shape[1]
    print(f'  Movie BoW features: {movie_features.shape}')

    # Train/test split (matching SeHGNN preprocessing)
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=int(0.1 * len(labels)),
        random_state=RAND_SEED,
    )
    train_idx, test_idx = train_test_split(
        train_idx,
        test_size=int(0.2 * len(labels)),
        random_state=RAND_SEED,
    )
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    # label.dat gets train+val, label.dat.test gets test
    train_val_idx = np.sort(np.concatenate([train_idx, val_idx]))
    print(f'  Splits: train+val={len(train_val_idx)}, test={len(test_idx)}')

    for var_id, cfg in VARIANT_EDGES.items():
        placement = ', '.join(
            f"{role}->{'+'.join(_normalize_targets(cfg[role]))}"
            for role in ROLES if role in cfg
        )
        print(f'\n--- Variant {var_id}: {placement} ---')
        save_dir = os.path.join(args.save_root, f'IMDB_var{var_id}{args.suffix}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        edges, rel_meta = build_edges(movies, directors, actors, imdb_links, years, cfg)
        print(f'  Edges: {len(edges)}, relation types: {len(rel_meta)}')

        write_node_dat(os.path.join(save_dir, 'node.dat'),
                       movies, directors, actors, imdb_links, years, movie_features)
        write_link_dat(os.path.join(save_dir, 'link.dat'), edges)
        write_label_dat(os.path.join(save_dir, 'label.dat'),
                        train_val_idx, movies, labels)
        write_label_dat(os.path.join(save_dir, 'label.dat.test'),
                        test_idx, movies, labels)
        write_info_dat(os.path.join(save_dir, 'info.dat'),
                       movies, directors, actors, imdb_links, years, rel_meta, feat_dim)

        np.save(os.path.join(save_dir, 'labels.npy'), labels)
        np.savez(os.path.join(save_dir, 'train_val_test_idx.npz'),
                 train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        print(f'  Saved to {save_dir}/')

    print('\nDone.')


if __name__ == '__main__':
    main()
