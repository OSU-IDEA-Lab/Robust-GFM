#!/bin/bash
# Frozen zero-shot evaluation of every variant (plus compiled views and a v1 repeat).
#   sbatch --export=ALL,DATASET=imdb,TASK=movie_director,DATA=rai-datasets/preprocessed/IMDB_movie_director,RUN_TAG=md_full slurm/zeroshot.sbatch
#SBATCH -J motif-zs
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tinw@oregonstate.edu

set -euo pipefail
: "${DATASET:?set DATASET}" "${TASK:?set TASK}" "${RUN_TAG:?set RUN_TAG}"
EXTRA=()
[[ -n ${DATA:-} ]] && EXTRA+=(--data "$DATA")
source slurm/env.sh
echo "dataset=$DATASET task=$TASK tag=$RUN_TAG"

python -u script/eval_lp.py --dataset "$DATASET" --task "$TASK" --run-tag "$RUN_TAG" "${EXTRA[@]}"
python script/report_lp.py --dataset "$DATASET" --run-tag "$RUN_TAG"
