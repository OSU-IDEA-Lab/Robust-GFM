#!/usr/bin/env bash
# PAIN x IMDb: exhaustive rooted simple paths (L = 3) on the year graph (CPU only; up to ~40 GB RAM).
# Input : data/raw/IMDB/movie_metadata.csv
# Output: data/preprocessed/pain_nc/IMDB/{shared.pt, v1..v4_L3.pt, universal_L3.pt}             (4.6 GB, ~10 min)
#         data/preprocessed/pain_nc/IMDB_invariant_minimal/{shared.pt, invariant_v1..v4_L3.pt}  (6.2 GB, ~7 min)
# The second step compiles each variant into the semantic program and checks that the four programs and
# path programs are byte-identical (status PASS in metadata.json).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/PAIN}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
python -m pain_nc.paths
python -m preprocessing.imdb_node_classification
python build_imdb_invariant_minimal.py
