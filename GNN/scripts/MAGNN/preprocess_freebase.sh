#!/bin/bash
# Build MAGNN's Freebase bundles data/preprocessed/freebase_magnn_v2/{invariant,invariant_union,stock}/<variant>/
# directly from the HGB-format variants data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/
# ({info,node,link,label}.dat + new_transitive_relations.jsonl registries; built by dataset_construction/).
# CPU only but memory-heavy: exact_3 / union_exact_2_3 link.dat are 8.5 / 9.3 GB of text -> use a large-memory node.
# ~2-10 min per bundle (~2 min invariant/unchanged ... ~10 min for the union forms).
# The semantic catalog = base relations + the shortcuts registered for the family {unchanged, exact_2, exact_3} (--family default).
# Split = the canonical RGCN partition (common/freebase_block_common.make_split, seed 1566911444).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RELEASE="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA="${INV_GNN_DATA:-$RELEASE/data}"; export INV_GNN_DATA="$DATA"
[ -n "${ENV_ACTIVATE:-}" ] && source "$ENV_ACTIVATE"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" PYTHONHASHSEED=0
cd "$RELEASE/models/MAGNN"
SRC="$DATA/dataset_variant_3hops_filtered"
DST="$DATA/preprocessed/freebase_magnn_v2"
for V in unchanged exact_2 exact_3; do
  python preprocess_freebase_magnn_v2.py --variants-root "$SRC" --variant "$V" --arm invariant --set-mode intersection --out "$DST/invariant"
  python preprocess_freebase_magnn_v2.py --variants-root "$SRC" --variant "$V" --arm invariant --set-mode union        --out "$DST/invariant_union"
done
for V in unchanged exact_2 exact_3 union_exact_2_3; do
  python preprocess_freebase_magnn_v2.py --variants-root "$SRC" --variant "$V" --arm stock --out "$DST/stock"
done
