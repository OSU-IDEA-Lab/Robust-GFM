#!/usr/bin/env bash
# RGCN, IMDb (year graph, canonical pooled-BoW features), task movie_director.
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] [DEVICE=cuda:0] bash scripts/RGCN/run_imdb_movie_director.sh
TASK=movie_director
source "$(dirname "${BASH_SOURCE[0]}")/_imdb.sh"
