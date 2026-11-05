#!/bin/bash
# CPU-only export of a node-classification dataset (Freebase exact_3 parses a 7.9 GB link.dat).
#   sbatch --export=ALL,DATASET=freebase slurm/nc_export.sbatch
#SBATCH -J motif-nc-export
#SBATCH -A eecs
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=64G  # exact_3 peaks around 40 GB
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tinw@oregonstate.edu

set -euo pipefail
: "${DATASET:?set DATASET}"
module load python/3.10
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -u script/nc_motif.py export --dataset "$DATASET" ${BUNDLE_ROOT:+--bundle-root "$BUNDLE_ROOT"}
