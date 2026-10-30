#!/usr/bin/env bash
# RGCN, IMDb (year graph, canonical pooled-BoW features), task movie_year.
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] [DEVICE=cuda:0] bash scripts/RGCN/run_imdb_movie_year.sh
TASK=movie_year
source "$(dirname "${BASH_SOURCE[0]}")/_imdb.sh"
