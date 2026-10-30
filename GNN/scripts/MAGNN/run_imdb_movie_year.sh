#!/bin/bash
# MAGNN x IMDb LP Movie-Year (year graph, v1-v4 + universal).
#   ARM=original|universal|augmentation|invariant  [FLAVOUR=union|intersection]  [SEEDS="1566911444 20241017 20251017"]
# Prerequisite: scripts/MAGNN/preprocess_imdb.sh. Resources: one GPU (V100 32 GB), 4 CPU cores, 48 GB RAM; ~10-25 min per run.
# Example: ARM=invariant FLAVOUR=union scripts/MAGNN/run_imdb_movie_year.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DATASET=IMDB; TASK=movie_year; VARIANTS="v1 v2 v3 v4"
BUNDLES="$DATA/preprocessed/imdb_magnn_v2"
OUT="$RESULTS/imdb_movie_year/$ARM"
RA="$(runner_arm)"
gauntlet "$BUNDLES/$TASK" "$VARIANTS" universal
source "$SCRIPT_DIR/_run_lp.sh"
