#!/bin/bash
# Build MAGNN's IMDb bundles data/preprocessed/imdb_magnn_v2/<task>/{invariant,invariant_union,stock}/<variant>/
# from the RGCN structural bundles data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/<task>/<variant>/graph.pt
# (built by the RGCN preprocessing -- see models/RGCN/README.md). CPU only, ~2-10 s per bundle; ~0.2 GB per bundle (features).
#   TASKS="nc movie_year movie_director" (default all three)
# Expected per-bundle summary.json program hashes are listed in models/MAGNN/README.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RELEASE="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA="${INV_GNN_DATA:-$RELEASE/data}"; export INV_GNN_DATA="$DATA"
[ -n "${ENV_ACTIVATE:-}" ] && source "$ENV_ACTIVATE"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" PYTHONHASHSEED=0
cd "$RELEASE/models/MAGNN"
SRC="$DATA/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB"
DST="$DATA/preprocessed/imdb_magnn_v2"
for TASK in ${TASKS:-nc movie_year movie_director}; do
  case "$TASK" in movie_director) VS="v1 v3";; *) VS="v1 v2 v3 v4";; esac
  for V in $VS; do
    python preprocess_imdb_magnn_v2.py --bundle-root "$SRC" --task "$TASK" --variant "$V" --arm invariant --set-mode intersection --out "$DST/$TASK/invariant"
    python preprocess_imdb_magnn_v2.py --bundle-root "$SRC" --task "$TASK" --variant "$V" --arm invariant --set-mode union        --out "$DST/$TASK/invariant_union"
  done
  for V in $VS universal; do
    python preprocess_imdb_magnn_v2.py --bundle-root "$SRC" --task "$TASK" --variant "$V" --arm stock --out "$DST/$TASK/stock"
  done
done
