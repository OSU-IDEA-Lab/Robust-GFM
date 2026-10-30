#!/usr/bin/env bash
# PathNet x Freebase: sample the walk sets (CPU only; one call per graph, all three seeds).
# 0.5-9 min per graph, up to ~50 GB RAM (union / exact_3).
# Input : RGCN's Freebase stores (bash scripts/RGCN/preprocess_freebase.sh):
#           data/preprocessed/freebase_rgcn_physical/<V>/physical_<V>/union/all_graph            (graph=physical)
#           data/preprocessed/freebase_rgcn_block_conditional/<V>/exact_3/union/all_graph        (graph=semantic)
# Output: data/preprocessed/pathnet_inv/freebase_walks/<graph>_<variant>_bfs/{walks_seed<s>.npz, meta.json}
# Expected: the three semantic walk sets have identical pair hashes (8ecdb0bdd522) and identical walks per seed.
#   GRAPHS="physical:unchanged ..." overrides the list below.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/PathNet}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
GRAPHS=${GRAPHS:-"physical:unchanged physical:exact_2 physical:exact_3 physical:union_exact_2_3 semantic:unchanged semantic:exact_2 semantic:exact_3"}
for gv in $GRAPHS; do
  python build_freebase_walks.py --graph "${gv%%:*}" --variant "${gv##*:}" --dist bfs
done
