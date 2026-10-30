#!/bin/bash
# SlotGAT x IMDb node classification (movie genre, 3 classes), year-bearing graphs IMDB_var{1..5}_year.
#   original     IMPL=upstream -> upstream SlotGAT (model/gnn.py), physical graph per variant
#                IMPL=parity   -> matched control: parity port, physical graph per variant
#                (IMPL defaults to "both" = both original rows)
#   universal    parity port on IMDB_var5_year (union graph), physical schedule
#   augmentation parity port, one model trained on var1-4
#   invariant    parity port, conditional (compiled semantic program)
# IMPL is read only for ARM=original. The upstream DGL kernels are not deterministic, so reruns of the
# IMPL=upstream row vary slightly.
# Usage: ARM=invariant [SEEDS="1566911444 20241017 20251017"] [GPU=0] bash scripts/SlotGAT/run_imdb_nc.sh
# Prerequisite: bash scripts/SlotGAT/preprocess_imdb.sh
# Resources: one GPU (tested on a V100 32 GB), 4 CPUs, 48 GB RAM; the whole cell takes under 1 h.
set -euo pipefail
ARM=${ARM:?set ARM=original|universal|augmentation|invariant}
SEEDS=${SEEDS:-"1566911444 20241017 20251017"}
IMPL=${IMPL:-both}
GPU=${GPU:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SEEDS_CSV=$(echo $SEEDS | tr ' ' ',')
OUT=results/imdb_nc/$ARM
CKPT=checkpoint/imdb_nc/$ARM
# Protocol shared by every IMDb-NC run.
COMMON="--data-suffix _year --feature-mode pooled_bow --select-on val_micro_f1 --epochs 300 --patience 40 --gpu $GPU"
case "$IMPL" in both|upstream|parity) ;; *) echo "unknown IMPL=$IMPL (both|upstream|parity)" >&2; exit 2 ;; esac
want () { [ "$IMPL" = both ] || [ "$IMPL" = "$1" ]; }
case "$ARM" in
  original)
    if want upstream; then
      python run_IMDB_nc_conditional.py --variants 1 2 3 4 --conditions original --seeds "$SEEDS_CSV" $COMMON \
        --results-dir "$OUT/upstream" --checkpoint-dir "$CKPT/upstream" --aggregate
    fi
    if want parity; then
      python run_IMDB_nc_conditional.py --variants 1 2 3 4 --conditions physical --seeds "$SEEDS_CSV" $COMMON \
        --model-impl parity --results-dir "$OUT/parity" --checkpoint-dir "$CKPT/parity" --aggregate
    fi ;;
  universal)
    python run_IMDB_nc_conditional.py --variants 5 --conditions physical --seeds "$SEEDS_CSV" $COMMON \
      --model-impl parity --results-dir "$OUT" --checkpoint-dir "$CKPT" --no-aggregate ;;
  augmentation)
    # no --select-on here: the augmentation runner uses its default selection criterion
    python run_IMDB_nc_augmentation.py --variants 1 2 3 4 --seeds "$SEEDS_CSV" \
      --feature-mode pooled_bow --data-suffix _year --epochs 300 --patience 40 --gpu "$GPU" \
      --model-impl parity --results-dir "$OUT" ;;
  invariant)
    python run_IMDB_nc_conditional.py --variants 1 2 3 4 --conditions conditional --seeds "$SEEDS_CSV" $COMMON \
      --model-impl parity --results-dir "$OUT" --checkpoint-dir "$CKPT" --aggregate
    # optional bitwise audit: VALIDATE=1
    if [ "${VALIDATE:-0}" = 1 ]; then
      python validate_IMDB_slotgat_conditional.py --variants 1 2 3 4 --gpu "$GPU" --feature-mode pooled_bow \
        --data-suffix _year --model-check --require-bitwise --output "$OUT/validation_model_audit.json"
    fi ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
echo "done: ARM=$ARM -> $(pwd)/$OUT"
