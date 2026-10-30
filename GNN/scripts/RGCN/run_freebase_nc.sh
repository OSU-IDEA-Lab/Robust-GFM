#!/usr/bin/env bash
# RGCN, Freebase node classification (BOOK genre, 8 classes), sampled 2-layer RGCN.
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] [DEVICE=cuda:0] bash scripts/RGCN/run_freebase_nc.sh
# invariant          = run_Freebase_rgcn_block_conditional.py on the exact_3 catalog stores
# original/universal = run_Freebase_rgcn_block_conditional.py on the physical-only stores
# augmentation       = run_FREEBASE_rgcn_augmentation_sampled.py (sampled joint training)
# Stores: data/preprocessed/freebase_rgcn_block_conditional/<V>/exact_3/union/all_graph and
#         data/preprocessed/freebase_rgcn_physical/<V>/physical_<V>/union/all_graph (scripts/RGCN/preprocess_freebase.sh).
# The invariance claim is bitwise equality across variants at a fixed seed: run the three variants
# of a seed on ONE GPU model (the script does).
# Resources: one V100 32 GB; ~10-25 min per run.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_arm
export PYTHONPATH="$RGCN${PYTHONPATH:+:$PYTHONPATH}"
cd "$RGCN"
COMMON_ARGS=(--set-mode union --scope all_graph --device "$DEVICE" --max-epochs 60 --patience 20
             --select-on val_micro_f1 --batch-size 64 --eval-batch-size 128
             --fanout-layer1 4 --fanout-layer2 4 --resume --skip-completed)
case "$ARM" in
  invariant)
    OUT="$RESULTS/freebase_nc/invariant"
    for SD in $SEEDS; do   # one seed at a time, all three variants
      python run_Freebase_rgcn_block_conditional.py \
        --preprocessed-root data/preprocessed/freebase_rgcn_block_conditional --results-root "$OUT" \
        --variants unchanged exact_2 exact_3 --seeds "$SD" --up-to exact_3 "${COMMON_ARGS[@]}"
    done ;;
  original|universal)
    OUT="$RESULTS/freebase_nc/$ARM"
    if [[ "$ARM" == universal ]]; then VS=union_exact_2_3; else VS="unchanged exact_2 exact_3"; fi
    for V in $VS; do
      python run_Freebase_rgcn_block_conditional.py \
        --preprocessed-root data/preprocessed/freebase_rgcn_physical --results-root "$OUT" \
        --variants "$V" --seeds $SEEDS --up-to "physical_$V" "${COMMON_ARGS[@]}"
    done ;;
  augmentation)
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    OUT="$RESULTS/freebase_nc/augmentation"
    python run_FREEBASE_rgcn_augmentation_sampled.py \
      --preprocessed-root data/preprocessed/freebase_rgcn_physical --results-root "$OUT" \
      --variants unchanged exact_2 exact_3 --seeds $SEEDS \
      --super-epochs 20 --patience 7 --device "$DEVICE" ;;
esac
echo "[done] Freebase $ARM -> $OUT"
