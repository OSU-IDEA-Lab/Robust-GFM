#!/bin/bash
# The unchanged checkpoint on the compiled view of every representation (built by scripts/build_data.sh).
# Every pair should give Kendall tau = 1.000 with bitwise-identical scores.
# Resources: as run_zero_shot.sh. Results: $RESULTS/compiled/<DATASET>/<CHECKPOINT>/.
#
#   bash scripts/run_compiled.sh Hetionet             # checkpoint defaults to ultra_4g
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DS=${1:?dataset: WN18RR | FB15k237 | Hetionet}
CK=${2:-ultra_4g}
REF=$(reference_of "$DS")
OUT="$RESULTS/compiled/$DS/$CK"
mkdir -p "$OUT"

for V in $(variants_of "$DS"); do
  [[ -f "$OUT/${V}__compiled/metrics.json" ]] && { echo "skip $V (done)"; continue; }
  python "$ULTRA/ultra_eval.py" --dataset-dir "$GFM_DATA/$DS" --variant "${V}__compiled" \
      --ckpt "$(ckpt_path "$CK")" --split test --out "$OUT/${V}__compiled"
done
python "$COMMON/compare_variants.py" --results "$OUT" --reference "${REF}__compiled"
