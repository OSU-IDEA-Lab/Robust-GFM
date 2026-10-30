#!/bin/bash
# SlotGAT x Freebase node classification (BOOK genre), family {unchanged, exact_2, exact_3},
# universal graph = union_exact_2_3. Parity port of upstream SlotGAT for every arm (--model-impl parity), lr 0.005.
#   original     --message-mode physical on each variant's own schedule
#   universal    --message-mode physical on union_exact_2_3
#   augmentation one shared model over the three physical schedules
#   invariant    --message-mode exact_union: every variant runs the compiled
#                union-message program (recoverability-conditional)
# Features/labels/splits always come from the `unchanged` source dir, so the universal arm passes
# --reference-data-dir pointing at it.
# Usage: ARM=invariant [SEEDS="..."] [GPU=0] [LR=0.005] bash scripts/SlotGAT/run_freebase_nc.sh
# Cost (V100): original ~1-12 h per seed (exact_3 dominates), universal/invariant ~6.5-9 min/epoch on the 750M-message
# union schedule, many hours per variant-run; runs are resumable (--resume), just rerun the script.
# Resources: one GPU, 4 CPUs, 32 GB RAM; 12 h time slices are enough with --resume.
# Prerequisite: bash scripts/SlotGAT/preprocess_freebase.sh
set -euo pipefail
ARM=${ARM:?set ARM=original|universal|augmentation|invariant}
SEEDS=${SEEDS:-"1566911444 20241017 20251017"}
GPU=${GPU:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=${INV_GNN_DATA:-data}
PLANS=$DATA/preprocessed/freebase_slotgat_block_family
PHYS=$DATA/preprocessed/freebase_slotgat_physical_messages_family
UNION=$DATA/preprocessed/freebase_slotgat_union_messages/union_exact_2_3
OUT=results/freebase_nc/$ARM
CKPT=checkpoint/freebase_nc/$ARM
TRAIN="--union-message-chunk-edges 250000 --select-on val_micro_f1 --stop-on val_loss --lr ${LR:-0.005} --model-impl parity \
  --epochs 150 --patience 25 --print-every 5 --resume --gpu $GPU"
case "$ARM" in
  original|universal)
    if [ "$ARM" = original ]; then VARS="unchanged exact_2 exact_3"; REF=""; else VARS="union_exact_2_3"; REF="--reference-data-dir $DATA/dataset_variant_3hops_filtered/unchanged"; fi
    for SEED in $SEEDS; do
      python run_Freebase_nc_block_conditional.py --preprocessed-root "$PLANS" --variants $VARS --seeds "$SEED" \
        --message-mode physical --physical-message-root "$PHYS" $REF $TRAIN \
        --results-dir "$OUT" --checkpoint-dir "$CKPT"
      python scripts/gate_freebase_runs.py --results-dir "$OUT" --seed "$SEED" --expect-mode physical --epoch-ceiling 150
    done ;;
  invariant)
    for SEED in $SEEDS; do
      python run_Freebase_nc_block_conditional.py --preprocessed-root "$PLANS" --variants unchanged exact_2 exact_3 --seeds "$SEED" \
        --message-mode exact_union --union-message-root "$UNION" $TRAIN \
        --results-dir "$OUT" --checkpoint-dir "$CKPT"
      python scripts/gate_freebase_runs.py --results-dir "$OUT" --seed "$SEED" --expect-mode exact_union --epoch-ceiling 150 --require-invariance
    done ;;
  augmentation)
    SEEDS_CSV=$(echo $SEEDS | tr ' ' ',')
    python run_Freebase_nc_augmentation.py --preprocessed-root "$PLANS" --physical-message-root "$PHYS" \
      --variants unchanged exact_2 exact_3 --lr "${LR:-0.005}" --model-impl parity --union-message-chunk-edges 250000 --gpu "$GPU" \
      --seeds "$SEEDS_CSV" --epochs 450 --patience 75 --print-every 5 --resume --results-dir "$OUT" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
echo "done: ARM=$ARM -> $(pwd)/$OUT"
