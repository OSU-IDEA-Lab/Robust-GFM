#!/bin/bash
# Zero-shot ULTRA on every representation of one dataset, then the per-pair comparison.
# The reference is evaluated a second time ("<reference>__repeat") to measure the noise floor.
# Finished variants are skipped, so re-running resumes.
# Resources: one GPU with >= 4 GB; 4 CPUs, 12 GB RAM. Time per (dataset, checkpoint) on an RTX 2080 Ti:
#   WN18RR ~10 min, Hetionet ~30 min, FB15k-237 ~1.5 h.
#
#   bash scripts/run_zero_shot.sh FB15k237 ultra_4g
#   LIMIT=64 bash scripts/run_zero_shot.sh WN18RR ultra_3g       # smoke test: first 64 queries
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DS=${1:?dataset: WN18RR | FB15k237 | Hetionet}
CK=${2:?checkpoint: ultra_3g | ultra_4g | ultra_50g}
LIMIT=${LIMIT:-0}
REF=$(reference_of "$DS")
OUT="$RESULTS/$DS/$CK"; [[ "$LIMIT" != 0 ]] && OUT="$RESULTS/smoke$LIMIT/$DS/$CK"
mkdir -p "$OUT"

for V in $(variants_of "$DS") "${REF}__repeat"; do
  [[ -f "$OUT/$V/metrics.json" ]] && { echo "skip $V (done)"; continue; }
  python "$ULTRA/ultra_eval.py" --dataset-dir "$GFM_DATA/$DS" --variant "${V%__repeat}" \
      --ckpt "$(ckpt_path "$CK")" --split test --out "$OUT/$V" --limit "$LIMIT"
done
python "$COMMON/compare_variants.py" --results "$OUT" --reference "$REF"
