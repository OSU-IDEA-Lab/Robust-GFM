#!/bin/bash
# Zero-shot evaluation on every representation of one dataset, then the per-pair comparison.
# The reference is evaluated a second time ("<reference>__repeat") to measure the noise floor.
# Finished variants are skipped, so re-running resumes. Resources: <fill in>.
#
#   bash scripts/run_zero_shot.sh WN18RR <checkpoint>
#   LIMIT=64 bash scripts/run_zero_shot.sh WN18RR <checkpoint>    # smoke test: first 64 queries
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/_common.sh"
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${RESULTS:-$MODEL_DIR/results}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DS=${1:?dataset: WN18RR | FB15k237 | Hetionet}
CK=${2:?checkpoint name (file under upstream/, without extension)}
CKPT="$MODEL_DIR/upstream/$CK"                 # adjust to where the checkpoints live
LIMIT=${LIMIT:-0}
REF=$(reference_of "$DS")
OUT="$RESULTS/$DS/$CK"; [[ "$LIMIT" != 0 ]] && OUT="$RESULTS/smoke$LIMIT/$DS/$CK"
mkdir -p "$OUT"

for V in $(variants_of "$DS") "${REF}__repeat"; do
  [[ -f "$OUT/$V/metrics.json" ]] && { echo "skip $V (done)"; continue; }
  python "$MODEL_DIR/evaluate.py" --dataset-dir "$GFM_DATA/$DS" --variant "${V%__repeat}" \
      --ckpt "$CKPT" --split test --out "$OUT/$V" --limit "$LIMIT"
done
python "$COMMON/compare_variants.py" --results "$OUT" --reference "$REF"
