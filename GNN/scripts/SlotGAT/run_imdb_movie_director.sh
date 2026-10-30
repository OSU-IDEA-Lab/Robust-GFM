#!/bin/bash
# SlotGAT x IMDb Movie-Director (MD) link prediction (models/SlotGAT/lp).
# Flags on top of the runner defaults: --mask-batch-edges --loss neg_logsigmoid --eval-every 1 --patience 20
# (each run also saves its full configuration in metrics.json).
#   ARM=original     -> --arm originals   (one model per variant, physical graph; variants v1 v3)
#   ARM=universal    -> --arm universal   (physical union graph)
#   ARM=augmentation -> --arm augmentation (one model, all variants' physical graphs)
#   ARM=invariant    -> --arm invariant   (per variant, compiled semantic program)
# Results: models/SlotGAT/results/lp_hetero/<originals|universal|augmentation|invariant>/IMDB/movie_director/<variant>/seed_<s>/;
#          aggregate with python lp/aggregate_lp_arms.py --family hetero_nls
# Inputs: RGCN bundles data/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/movie_director/<variant>/graph.pt (built by the RGCN preprocessing scripts).
# Cost: minutes per run on one V100 (the whole 4-arm x 3-seed cell takes well under 4 h).
# Usage: ARM=invariant [SEEDS="1566911444 20241017 20251017"] bash scripts/SlotGAT/run_imdb_movie_director.sh
set -euo pipefail
ARM=${ARM:?set ARM=original|universal|augmentation|invariant}
SEEDS=${SEEDS:-"1566911444 20241017 20251017"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=${OUT_ROOT:-results/lp_hetero}
EXTRA="--mask-batch-edges --loss neg_logsigmoid --eval-every 1 --patience 20"
case "$ARM" in original) A=originals ;; universal|augmentation|invariant) A=$ARM ;; *) echo "unknown ARM=$ARM" >&2; exit 2 ;; esac
for SEED in $SEEDS; do
  R="python lp/run_hetero_lp_arms.py --dataset IMDB --task movie_director --out-root $OUT --seed $SEED $EXTRA"
  case $A in
    originals|invariant) for V in v1 v3; do $R --arm $A --variant $V; done ;;
    *) $R --arm $A ;;
  esac
done
echo "done: ARM=$ARM -> $(pwd)/$OUT/$A/IMDB/movie_director"
