#!/usr/bin/env bash
# PathNet x IMDb node classification (year graph, canonical pooled-BoW features), v1-v4.
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="1566911444 20241017 20251017"] bash scripts/PathNet/run_imdb_nc.sh
# Prerequisite: data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/nc/{v1..v4,universal}/graph.pt
#               (bash scripts/RGCN/preprocess_imdb.sh). Walks are sampled inside the runner; no PathNet preprocessing.
# Protocol (upstream PathNet_homo): fresh MERW walks every epoch (40 x 4, exact BFS distances), 1000 epochs without
# early stopping, lr 5e-3, dropout .7, weight decay 5e-4, hidden 64; the best epoch is selected on val micro-F1.
# Resources: one GPU (V100 32 GB), 4 CPU cores, 32 GB RAM; ~6 min per run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT=${OUT:-results/imdb_nc}
HP="--model pathnet_homo --resample --walk-type merw --lr 5e-3 --dropout 0.7 --epochs 1000 --patience 0 --hidden 64 --dist bfs --out-root $OUT"
for s in $SEEDS; do
  case "$ARM" in
    original|invariant) for v in v1 v2 v3 v4; do python run_pathnet_imdb_nc.py --arm $RUN_ARM --variant $v --seed $s $HP; done ;;
    universal|augmentation) python run_pathnet_imdb_nc.py --arm $RUN_ARM --seed $s $HP ;;
  esac
done
echo "summary: python summarize_arms.py $OUT IMDB/nc   (run from models/PathNet)"
