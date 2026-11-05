#!/bin/bash
# Fine-tuning matrix for the lab's three training seeds, one array task per seed.
#   sbatch slurm/finetune_seeds.sbatch                      # IMDb movie_year
#   sbatch --export=ALL,DATASET=dblp slurm/finetune_seeds.sbatch
#   sbatch --export=ALL,TASK=movie_director,DATA=rai-datasets/preprocessed/IMDB_movie_director slurm/finetune_seeds.sbatch
#   sbatch -p dgx2 --constraint=v100 slurm/finetune_seeds.sbatch   # any GPU; arch is detected in env.sh
# Keep all seeds of one table on the same GPU type (the CMPNN baselines ran on V100).
# Then build the seed-averaged tables with script/report_seeds.py (see docs/SEED_TABLES.md).
#SBATCH -J motif-ft-seeds
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00  # IMDb/DBLP matrices take <20 min; short limits avoid the per-user CPU-minutes cap
#SBATCH --array=0-2
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tinw@oregonstate.edu

set -euo pipefail
SEEDS=(1566911444 20241017 20251017)  # SCR METRICS.md seeds, shared with the CMPNN results
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASET:-imdb}
RUN_TAG=${RUN_TAG:-seeds}
EXTRA=()
[[ -n ${TASK:-} ]] && EXTRA+=(--task "$TASK")
[[ -n ${DATA:-} ]] && EXTRA+=(--data "$DATA")
source slurm/env.sh
echo "dataset=$DATASET task=${TASK:-default} seed=$SEED tag=$RUN_TAG"

if [[ ${SMOKE:-0} == 1 ]]; then  # Two updates on v1: checks the training path before the full array.
  python -u script/finetune_${DATASET}_lp.py "${EXTRA[@]}" --train-variant v1 --seed "$SEED" \
    --steps 2 --validate-every 2 --val-limit 16 --run-tag "smoke_${SLURM_JOB_ID}"
  exit
fi

python -u script/run_finetune_matrix.py "${EXTRA[@]}" \
  --dataset "$DATASET" --seed "$SEED" --run-tag "$RUN_TAG" \
  --steps 1000 --validate-every 100 --resume
