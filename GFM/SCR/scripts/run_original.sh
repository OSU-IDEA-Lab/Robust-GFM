#!/bin/bash
# Original-arm SCR only: DBLP paper-venue, IMDb node classification, IMDb
# movie-director, and IMDb movie-year. Zero-shot, then fine-tune. No invariant
# graphs and no ULTRA.
#
#   export MAGNN_ROOT=/path/to/MAGNN/data/preprocessed
#   export IMDB_CSV=/path/to/movie_metadata.csv
#   export IMDB_ML_SPLIT=/path/to/IMDB_ml_shared_splits.npz
#   bash scripts/run_original.sh
#
# One dataset at a time. Two SCR jobs on the same dataset collide on processed/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
: "${MAGNN_ROOT:?set MAGNN_ROOT to the MAGNN preprocessed directory}"
: "${IMDB_CSV:?set IMDB_CSV to movie_metadata.csv}"
: "${IMDB_ML_SPLIT:?set IMDB_ML_SPLIT to IMDB_ml_shared_splits.npz}"
CKPT="${CKPT:-$HERE/upstream/ckpts/scr_3g.pth}"
[[ -f "$CKPT" ]] || { echo "missing $CKPT; run bash setup_upstream.sh" >&2; exit 1; }

python preprocess/build_dblp_lp.py --magnn-root "$MAGNN_ROOT"
python preprocess/build_imdb_nc.py --magnn-root "$MAGNN_ROOT"
python preprocess/build_imdb_md.py --magnn-root "$MAGNN_ROOT"
python preprocess/build_imdb_my.py --magnn-root "$MAGNN_ROOT" --csv "$IMDB_CSV" --split "$IMDB_ML_SPLIT"

run_one() {
  local dataset="$1" setting="$2" seed="$3" extra="$4"
  local out="results/${setting}/scr_3g/${dataset}/seed${seed}"
  if [[ -f "$out/metrics.json" ]]; then
    echo "skip $out"
    return
  fi
  if [[ "$setting" == "zs" ]]; then
    python nc_zero_shot.py --dataset "$dataset" --ckpt "$CKPT" --seed "$seed" --out "$out"
  else
    python nc_finetune.py --dataset "$dataset" --ckpt "$CKPT" --seed "$seed" --out "$out" $extra
  fi
}

NC_SEEDS=(1024 42 1337)
LP_SEEDS=(1566911444 20241017 20251017)

for v in v1 v2 v3 v4; do
  for s in "${NC_SEEDS[@]}"; do
    run_one "IMDB_NC_${v}" zs "$s" ""
    run_one "IMDB_NC_${v}" ft "$s" "--epochs 5 --select acc"
  done
done

for v in v1 v2 v3; do
  for s in "${LP_SEEDS[@]}"; do
    run_one "DBLP_LP_${v}" zs "$s" ""
    run_one "DBLP_LP_${v}" ft "$s" "--epochs 1 --select mrr"
  done
done

for v in v1 v3; do
  for s in "${LP_SEEDS[@]}"; do
    run_one "IMDB_MD_${v}" zs "$s" ""
    run_one "IMDB_MD_${v}" ft "$s" "--epochs 5 --select mrr"
  done
done

for v in v1 v2 v3 v4; do
  for s in "${NC_SEEDS[@]}"; do
    run_one "IMDB_MY_${v}" zs "$s" ""
    run_one "IMDB_MY_${v}" ft "$s" "--epochs 5 --select mrr"
  done
done

echo "ALL DONE"
