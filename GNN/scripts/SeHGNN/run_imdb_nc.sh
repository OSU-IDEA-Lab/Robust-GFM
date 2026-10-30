#!/bin/bash
# SeHGNN x IMDb node classification (year graph, v1-v4), one arm x seeds, channel set `full`
# ("full-K, relation-aware" = one channel per type path, K=4; invariant: skip schema, Link context-only, 25 channels).
#
#   ARM      = original | universal | augmentation | invariant               (required)
#   SEEDS    = "1566911444 20241017 20251017" (default)
#   OUT      = output root (default models/SeHGNN/results); runs land in
#              $OUT/IMDB_nc/<armdir>_full/IMDB/nc/<variant>/seed_<s>/{metrics.json,scores.npz,trace.json,checkpoint.pt}
#              with <armdir> = originals | universal | augmentation | invariant (the runner's internal arm names).
#   AGGREGATE=1 also writes <armdir>_full/aggregate/ (Kendall tau etc.; needs common/aggregate_rgcn_structural.py)
#   EXTRA    = extra runner flags (none needed for the documented results)
# Needs: bash scripts/SeHGNN/preprocess_imdb.sh
# Resources: one GPU (V100 32 GB), 4 CPU cores, 48 GB RAM; ~1-4 min per run.
# Example: ARM=invariant bash scripts/SeHGNN/run_imdb_nc.sh
set -uo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
${ENV_ACTIVATE:-}
export DGLBACKEND=pytorch OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export INV_GNN_DATA="${INV_GNN_DATA:-$REL/data}"
cd "$REL/models/SeHGNN"
ARM="${ARM:?set ARM=original|universal|augmentation|invariant}"
SEEDS="${SEEDS:-1566911444 20241017 20251017}"
OUT="${OUT:-results}"
EXTRA="${EXTRA:-}"
DATA="$INV_GNN_DATA/preprocessed/sehgnn_v2"
VARIANTS="v1 v2 v3 v4"
FL=full
FAILED=0
for SD in $SEEDS; do
  case "$ARM" in
    original|invariant)
      A=$([ "$ARM" = original ] && echo originals || echo invariant)
      for V in $VARIANTS; do
        python run_sehgnn_nc.py --dataset IMDB --arm "$A" --flavor "$FL" --variant "$V" --seed "$SD" --data-root "$DATA" \
          --output-root "$OUT" --resume $EXTRA || FAILED=1
      done ;;
    universal)
      python run_sehgnn_nc.py --dataset IMDB --arm universal --flavor "$FL" --variant universal --seed "$SD" --data-root "$DATA" \
        --output-root "$OUT" --resume $EXTRA || FAILED=1 ;;
    augmentation)
      python run_sehgnn_nc_augmentation.py --dataset IMDB --flavor "$FL" --variants v1,v2,v3,v4 --seed "$SD" --data-root "$DATA" \
        --output-root "$OUT" --resume $EXTRA || FAILED=1 ;;
    *) echo "unknown ARM=$ARM"; exit 2 ;;
  esac
done
if [ "${AGGREGATE:-0}" = 1 ]; then
  A=$(case "$ARM" in original) echo originals;; *) echo "$ARM";; esac)
  D="$OUT/IMDB_nc/${A}_${FL}"
  python ../../common/aggregate_rgcn_structural.py --results-root "$D" --out-dir "$D/aggregate" | tail -3
fi
exit $FAILED
