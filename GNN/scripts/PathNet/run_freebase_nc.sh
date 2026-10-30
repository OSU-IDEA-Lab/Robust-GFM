#!/usr/bin/env bash
# PathNet x Freebase node classification (unchanged, exact_2, exact_3; universal = union_exact_2_3).
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] bash scripts/PathNet/run_freebase_nc.sh
# Prerequisite: bash scripts/PathNet/preprocess_freebase.sh (BFS walk sets; they provide the targets and hashes).
# Protocol (upstream PathNet_homo, as on IMDb): fresh MERW walks every epoch (40 x 4, exact BFS distances),
# 1000 epochs without early stopping, lr 5e-3, dropout .7, weight decay 5e-4; the best epoch is selected on val
# micro-F1. Input = fixed seeded random node vectors + learned node-type embedding (--input rand_type), every arm.
# The first run on each graph builds an exact 2-hop bitset (~180 MB, up to ~30 min on exact_3 / union) and caches it
# in data/preprocessed/pathnet_inv/freebase_twohop_bits/; the invariant arm builds one bitset for all three variants.
# Resources: one GPU (V100 32 GB), 4 CPU cores, ~70 GB RAM; 25-45 min per run once the bitsets exist.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT=${OUT:-results/freebase_nc}
HP="--input rand_type --walk-suffix _bfs --model pathnet_homo --resample --walk-type merw --lr 5e-3 --dropout 0.7 --wd 5e-4 --epochs 1000 --patience 0 --out-root $OUT"
for s in $SEEDS; do
  case "$ARM" in
    original|invariant) for v in unchanged exact_2 exact_3; do python run_pathnet_freebase_nc.py --arm $RUN_ARM --variant $v --seed $s $HP; done ;;
    universal|augmentation) python run_pathnet_freebase_nc.py --arm $RUN_ARM --seed $s $HP ;;
  esac
done
echo "summary: python summarize_arms.py $OUT FREEBASE/nc   (run from models/PathNet)"
