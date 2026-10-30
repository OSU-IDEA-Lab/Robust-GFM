#!/bin/bash
# Runs ULTRA's own script/run.py zero-shot on its own copy of WN18RR (downloaded by upstream on first use),
# to check that ultra_eval.py reproduces upstream: the test MRR printed here must equal the pessimistic
# MRR of the WN18RR reference in run_zero_shot.sh (0.4838 for ultra_4g, 0.3691 for ultra_3g).
# Resources: one GPU, 32 GB RAM, ~15 min. Outputs: $RESULTS/ref_check/<CHECKPOINT>/.
#
#   bash scripts/run_ref_check.sh ultra_4g
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CK=${1:?checkpoint: ultra_3g | ultra_4g | ultra_50g}
OUT="$RESULTS/ref_check/$CK"
mkdir -p "$OUT/output" "$OUT/kg-datasets"
# upstream writes working_dir.tmp into the current directory: one directory per checkpoint
cd "$OUT"
python "$ULTRA/ref_check.py" -c "$ULTRA/ref_check_inference.yaml" --dataset WN18RR --epochs 0 --bpe null \
    --gpus "[0]" --ckpt "$(ckpt_path "$CK")" --output_root "$OUT/output" --dataset_root "$OUT/kg-datasets"
