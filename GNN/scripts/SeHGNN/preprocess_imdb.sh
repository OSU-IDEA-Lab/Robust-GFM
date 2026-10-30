#!/bin/bash
# SeHGNN x IMDb (year graph): compile the channel bundles read by the four arms (channel set `full`, full-K;
# invariant = skip schema, Link context-only, 25 channels), then run the data-level gauntlet.
#
#   prerequisite : data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/nc/{v1,v2,v3,v4,universal}/graph.pt
#                  (bash scripts/RGCN/preprocess_imdb.sh, see models/RGCN/README.md)
#   output       : data/preprocessed/sehgnn_v2/IMDB/full/{semantic/{v1..v4}, physical/{v1..v4,universal}}/
# CPU only, seconds to ~1 min per bundle, ~8 GB RAM. Usage:  bash scripts/SeHGNN/preprocess_imdb.sh
set -euo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
${ENV_ACTIVATE:-}
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONHASHSEED=0
export INV_GNN_DATA="${INV_GNN_DATA:-$REL/data}"
cd "$REL/models/SeHGNN"
OUT="$INV_GNN_DATA/preprocessed/sehgnn_v2"
for v in v1 v2 v3 v4; do
  # invariant arm (semantic program; skip schema, Link context-only -- the default --skip-types link)
  python preprocess_sehgnn_channels.py --dataset IMDB --variant "$v" --graph semantic --flavor full --out-root "$OUT"
  # original arm (+ augmentation reads the same physical bundles)
  python preprocess_sehgnn_channels.py --dataset IMDB --variant "$v" --graph physical --flavor full --out-root "$OUT"
done
# universal arm (union graph)
python preprocess_sehgnn_channels.py --dataset IMDB --variant universal --graph physical --flavor full --out-root "$OUT"
# data-level gauntlet (physical hashes differ, program/supervision/features equal across v1-v4)
python validate_sehgnn_invariance.py --dataset IMDB --flavors full --data-root "$OUT" --skip-first-step \
  --out "$OUT/IMDB/gauntlet.json"
