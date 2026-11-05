#!/bin/bash
# MOTIF node classification (script/nc_motif.py) on the IMDb (v1-v4) or Freebase (v1-v3) variants.
#   Zero-shot, one task per variant:
#     sbatch --array=0-3 --export=ALL,DATASET=imdb,MODE=eval slurm/nc.sbatch
#   Fine-tune then evaluate, one task per (variant, lab seed); task i -> variant i/3, seed i%3:
#     sbatch --array=0-11 --export=ALL,DATASET=imdb,MODE=finetune slurm/nc.sbatch
#   Smoke tests: ARGS="--limit 16" (eval) or ARGS="--steps 2 --val-limit 16" (finetune) and a fresh RUN_TAG.
#   Freebase v3 (447M edges) overflows cuSPARSE in the hypergraph lift: add SCRIPT=script/nc_motif_dense.py.
#SBATCH -J motif-nc
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00  # short limits keep array tasks under the per-user CPU-minutes cap
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tinw@oregonstate.edu

set -euo pipefail
: "${MODE:?set MODE=eval or MODE=finetune}" "${DATASET:?set DATASET=imdb or DATASET=freebase}"
VARIANTS=(v1 v2 v3 v4)
SEEDS=(1566911444 20241017 20251017)  # SCR METRICS.md seeds, shared with the lab baselines
RUN_TAG=${RUN_TAG:-seeds}
SCRIPT=${SCRIPT:-script/nc_motif.py}
read -r -a EXTRA <<< "${ARGS:-}"
source slurm/env.sh

if [[ $MODE == eval ]]; then
  VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
  echo "zero-shot dataset=$DATASET variant=$VARIANT tag=$RUN_TAG"
  python -u "$SCRIPT" eval --dataset "$DATASET" --variant "$VARIANT" --run-tag "$RUN_TAG" "${EXTRA[@]}"
  exit
fi

VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "fine-tune dataset=$DATASET variant=$VARIANT seed=$SEED tag=$RUN_TAG"
python -u "$SCRIPT" finetune --dataset "$DATASET" --variant "$VARIANT" --seed "$SEED" --run-tag "$RUN_TAG" "${EXTRA[@]}"
CKPT=results/motif_finetune_nc_${DATASET}_${RUN_TAG}/train_${VARIANT}/seed${SEED}/best.pth
LIMIT=()
[[ " ${EXTRA[*]} " == *" --val-limit 16 "* ]] && LIMIT=(--limit 16)  # Smoke runs evaluate a small slice too.
python -u "$SCRIPT" eval --dataset "$DATASET" --variant "$VARIANT" --checkpoint "$CKPT" --run-tag "$RUN_TAG" "${LIMIT[@]}"
