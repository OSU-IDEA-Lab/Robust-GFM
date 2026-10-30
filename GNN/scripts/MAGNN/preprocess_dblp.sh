#!/bin/bash
# Build MAGNN's DBLP bundles data/preprocessed/dblp_magnn_v2/{invariant,invariant_union,stock}/<variant>/
# from the RGCN structural bundles data/preprocessed/RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}/graph.pt
# (built by the RGCN preprocessing -- see models/RGCN/README.md). CPU only, ~10-20 s per bundle.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RELEASE="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA="${INV_GNN_DATA:-$RELEASE/data}"; export INV_GNN_DATA="$DATA"
[ -n "${ENV_ACTIVATE:-}" ] && source "$ENV_ACTIVATE"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" PYTHONHASHSEED=0
cd "$RELEASE/models/MAGNN"
SRC="$DATA/preprocessed/RGCN_structural_v8_dblp/DBLP"
DST="$DATA/preprocessed/dblp_magnn_v2"
for V in v1 v2 v3; do
  python preprocess_dblp_magnn_v2.py --bundle-root "$SRC" --variant "$V" --arm invariant --set-mode intersection --out "$DST/invariant"
  python preprocess_dblp_magnn_v2.py --bundle-root "$SRC" --variant "$V" --arm invariant --set-mode union        --out "$DST/invariant_union"
done
for V in v1 v2 v3 universal; do
  python preprocess_dblp_magnn_v2.py --bundle-root "$SRC" --variant "$V" --arm stock --out "$DST/stock"
done
