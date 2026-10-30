#!/usr/bin/env bash
# RGCN, WN18RR link prediction (filtered MRR over all 40,943 entities).
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] [DEVICE=cuda:0] bash scripts/RGCN/run_wn18rr_lp.sh
# invariant           = run_wordnet_rgcn_structural_full.sh (semantic program, task_only objective)
# original, universal = run_WORDNET_rgcn_structural.py --graph physical
# augmentation        = augmentation/run_WORDNET_rgcn_augmentation.py
# Bundle: data/preprocessed/RGCN_structural/WORDNET/universal_parity_full (scripts/RGCN/preprocess_wn18rr.sh).
# Long runs: up to 8000 epochs (invariant / originals ~45-70 min per run on a V100; augmentation ~6 h per seed).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_arm
WN_RAW="data/wordnet_3hops_augmented_full"
BUNDLE="data/preprocessed/RGCN_structural/WORDNET/universal_parity_full"
OUT="$RESULTS/wn18rr_lp/$ARM"
VARIANTS="${VARIANTS:-no_changes all_inverse_edges transitive_edges}"
case "$ARM" in
  invariant)
    cd "$RGCN"
    PYTHON_BIN=python DEVICE="$DEVICE" SEEDS="$SEEDS" VARIANTS="$VARIANTS" \
    BUDGETS_TO_RUN=full MODES_TO_RUN=intersection TRAIN_OBJECTIVE=task_only \
    DATASET_ROOT="$RGCN/$WN_RAW" PREPROCESSED_ROOT="$RGCN/$BUNDLE" RESULTS_ROOT="$OUT" \
    MAX_ENDPOINTS_PER_SOURCE=0 EPOCHS=8000 PATIENCE=500 \
    RUN_TESTS=0 PREPARE_SPLITS=0 PREPROCESS=0 VALIDATE=0 RUN_PARITY_DIAGNOSTIC=0 \
    TRAIN=1 AGGREGATE=1 SKIP_COMPLETED=1 RESUME_MODE=auto \
    bash run_wordnet_rgcn_structural_full.sh ;;
  original|universal)
    cd "$RGCN"
    if [[ "$ARM" == universal ]]; then VS=universal_edges; else VS="$VARIANTS"; fi
    for V in $VS; do for SD in $SEEDS; do
      python run_WORDNET_rgcn_structural.py \
        --data-root "$BUNDLE" --output-root "$OUT" \
        --budget full --mode intersection --variant "$V" --seed "$SD" \
        --device "$DEVICE" --graph physical --train-objective task_only \
        --epochs 8000 --patience 500 --eval-interval 10
    done; done ;;
  augmentation)
    cd "$AUG"
    for SD in $SEEDS; do
      [[ -s "$OUT/seed_$SD/summary.json" ]] && { echo "[skip] seed $SD complete"; continue; }
      python run_WORDNET_rgcn_augmentation.py \
        --variants no_changes,all_inverse_edges,transitive_edges \
        --seeds "$SD" \
        --splits-path "$WN_RAW/wordnet_splits.npz" \
        --super-epochs 2667 --patience 167 \
        --resume \
        --output-dir "$OUT"
    done ;;
esac
echo "[done] WN18RR $ARM -> $OUT"
