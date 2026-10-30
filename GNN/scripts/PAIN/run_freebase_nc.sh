#!/usr/bin/env bash
# PAIN x Freebase node classification (unchanged, exact_2, exact_3; universal = union_exact_2_3).
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] bash scripts/PAIN/run_freebase_nc.sh
# Prerequisite: bash scripts/PAIN/preprocess_freebase.sh
# Settings: 1 PAIN layer, otherwise configs/freebase_nc.yaml (learned node-id + node-type embeddings);
# <= 100 epochs, patience 20. Runs resume from a per-epoch state file.
# Resources: one GPU (V100 32 GB, ~4 GB used), 4 CPU cores, 96 GB RAM; 1.5-3 h per run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT=${OUT:-results/freebase_nc}
HP="--dataset FREEBASE --num-layers 1 --epochs 100 --patience 20 --out-root $OUT"
for s in $SEEDS; do
  case "$ARM" in
    original|invariant) for v in unchanged exact_2 exact_3; do python run_pain_nc.py --arm $RUN_ARM --variant $v --seed $s $HP; done ;;
    universal|augmentation) python run_pain_nc.py --arm $RUN_ARM --seed $s $HP ;;
  esac
done
echo "summary: python summarize_arms.py $OUT FREEBASE/nc   (run from models/PAIN)"
