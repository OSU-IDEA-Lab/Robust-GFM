#!/bin/bash
# WN18RR variants (v1 no_changes, v2 all_inverse_edges, v3 transitive_edges) with script/wordnet_lp.py.
#   Zero-shot, one task per variant:
#     sbatch --array=0-2 --export=ALL,MODE=eval slurm/wordnet.sbatch
#   Fine-tune then evaluate, one task per (variant, lab seed); task i -> variant i/3, seed i%3:
#     sbatch --array=0-8 --export=ALL,MODE=finetune slurm/wordnet.sbatch
#   Smoke tests: add ARGS="--limit 16" (eval) or ARGS="--steps 2 --val-limit 16" (finetune) and a fresh RUN_TAG.
#SBATCH -J motif-wn
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00  # short limits keep array tasks under the per-user CPU-minutes cap
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tinw@oregonstate.edu

set -euo pipefail
: "${MODE:?set MODE=eval or MODE=finetune}"
VARIANTS=(v1 v2 v3)
SEEDS=(1566911444 20241017 20251017)  # SCR METRICS.md seeds, shared with the CMPNN results
RUN_TAG=${RUN_TAG:-seeds}
read -r -a EXTRA <<< "${ARGS:-}"
source slurm/env.sh

if [[ $MODE == eval ]]; then
  VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
  echo "zero-shot variant=$VARIANT tag=$RUN_TAG"
  python -u script/wordnet_lp.py eval --variant "$VARIANT" --run-tag "$RUN_TAG" "${EXTRA[@]}"
  exit
fi

VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "fine-tune variant=$VARIANT seed=$SEED tag=$RUN_TAG"
python -u script/wordnet_lp.py finetune --variant "$VARIANT" --seed "$SEED" --run-tag "$RUN_TAG" "${EXTRA[@]}"
CKPT=results/motif_finetune_wordnet_lp_${RUN_TAG}/train_${VARIANT}/seed${SEED}/best.pth
LIMIT=()
[[ " ${EXTRA[*]} " == *" --val-limit 16 "* ]] && LIMIT=(--limit 16)  # Smoke runs evaluate a small slice too.
python -u script/wordnet_lp.py eval --variant "$VARIANT" --checkpoint "$CKPT" --run-tag "$RUN_TAG" "${LIMIT[@]}"
