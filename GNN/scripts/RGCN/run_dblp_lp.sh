#!/usr/bin/env bash
# RGCN, DBLP link prediction Paper--Conference (20-way full ranking).
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] [DEVICE=cuda:0] bash scripts/RGCN/run_dblp_lp.sh
# original/universal/invariant = run_rgcn_structural.py on the bundle
#   data/preprocessed/RGCN_structural_v8_dblp (scripts/RGCN/preprocess_dblp.sh).
# augmentation = augmentation/run_DBLP_rgcn_augmentation.py on the stores
#   data/preprocessed/DBLP_rgcn_v{1,2,3}_canonical (scripts/RGCN/preprocess_dblp.sh).
# Resources: one V100 32 GB; ~1 min per run (original/universal/invariant), ~4 min per seed (augmentation).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_arm
BUNDLE="${BUNDLE:-data/preprocessed/RGCN_structural_v8_dblp}"
OUT="$RESULTS/dblp_lp/$ARM"
run_one () {  # $1 = --graph, $2 = variant
  for SD in $SEEDS; do
    python run_rgcn_structural.py \
      --data-root "$BUNDLE" --output-root "$OUT" \
      --dataset DBLP --task paper_conference --variant "$2" --seed "$SD" \
      --device "$DEVICE" --graph "$1" --decoder dot --loss pairwise \
      --select-on val_mrr --norm mean --eval-protocol full_target_type \
      --emb-init 0.1 --epochs 1000 --patience 200
  done
}
case "$ARM" in
  invariant) cd "$RGCN"; for V in v1 v2 v3; do run_one semantic "$V"; done ;;
  original)  cd "$RGCN"; for V in v1 v2 v3; do run_one physical "$V"; done ;;
  universal) cd "$RGCN"; run_one physical universal ;;
  augmentation)
    # the augmentation arm uses 4 BLAS threads
    export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    cd "$AUG"
    python run_DBLP_rgcn_augmentation.py --variants v1,v2,v3 --seeds "$(comma $SEEDS)" \
      --data-suffix _canonical --select-on val_mrr --neg-per-paper 0 \
      --super-epochs 334 --patience 67 \
      --output-dir "$OUT" ;;
esac
echo "[done] DBLP $ARM -> $OUT"
