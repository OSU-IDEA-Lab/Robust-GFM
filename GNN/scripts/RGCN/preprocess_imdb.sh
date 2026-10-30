#!/usr/bin/env bash
# RGCN IMDb structural bundles (year graph, canonical pooled-BoW features) -- SHARED PREREQUISITE:
# also read by CMPNN (MY / MD), MAGNN, SeHGNN and SlotGAT (LP).
# Input : data/raw/IMDB/movie_metadata.csv
#         data/preprocessed/IMDB_shared_splits/IMDB_{md,my}_shared_splits.npz   (dataset_construction/)
# Output: data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/{nc/{v1..v4,universal},
#         movie_year/{v1..v4,universal}, movie_director/{v1,v3,universal}}/{graph.pt,audit.json}  (~2.4 GB)
# CPU only, a few minutes. Expected semantic hashes: nc 942317043f15, movie_director 63298526d97b,
# movie_year d74dc76282a9 (printed by the script; also in each audit.json).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RGCN"
python preprocess_IMDB_rgcn_structural.py \
  --tasks nc,movie_director,movie_year \
  --raw-csv data/raw/IMDB/movie_metadata.csv \
  --shared-movie-director data/preprocessed/IMDB_shared_splits/IMDB_md_shared_splits.npz \
  --shared-movie-year data/preprocessed/IMDB_shared_splits/IMDB_my_shared_splits.npz \
  --out-root data/preprocessed/RGCN_structural_v10_year_canonfeat
