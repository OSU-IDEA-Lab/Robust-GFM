#!/usr/bin/env bash
# PAIN x IMDb node classification (year graph, canonical pooled-BoW features), v1-v4.
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="1566911444 20241017 20251017"] bash scripts/PAIN/run_imdb_nc.sh
# Prerequisite: bash scripts/PAIN/preprocess_imdb.sh
# Settings: 1 PAIN layer (selected on the original arm), otherwise configs/imdb_nc.yaml; <= 300 epochs, patience 40.
# Runs resume from a per-epoch state file, so an interrupted run can simply be restarted.
# Resources: one GPU (V100 32 GB, ~6 GB used), 4 CPU cores, 96 GB RAM; 15-50 min per run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT=${OUT:-results/imdb_nc}
HP="--dataset IMDB --num-layers 1 --out-root $OUT"
for s in $SEEDS; do
  case "$ARM" in
    original|invariant) for v in v1 v2 v3 v4; do python run_pain_nc.py --arm $RUN_ARM --variant $v --seed $s $HP; done ;;
    universal|augmentation) python run_pain_nc.py --arm $RUN_ARM --seed $s $HP ;;
  esac
done
echo "summary: python summarize_arms.py $OUT IMDB/nc   (run from models/PAIN)"
