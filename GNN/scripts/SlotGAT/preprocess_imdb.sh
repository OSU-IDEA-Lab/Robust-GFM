#!/bin/bash
# SlotGAT IMDb NC inputs: writes data/IMDB_var{1..5}_year/{node,link,label,label.dat.test,info}.dat + labels.npy
# + train_val_test_idx.npz from data/raw/IMDB/movie_metadata.csv (CPU, ~1-2 min, ~575 MB written).
# Deterministic: a rebuild gives byte-identical files.
# The SlotGAT LP cells (IMDb MY/MD) do NOT use these: they consume RGCN's
# data/preprocessed/RGCN_structural_v10_year_canonfeat bundles (build them with the RGCN scripts first).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export PYTHONHASHSEED=0
python preprocess_IMDB.py --suffix _year ${SAVE_ROOT:+--save-root "$SAVE_ROOT"}
