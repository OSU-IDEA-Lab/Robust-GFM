# Shared setup for the PAIN scripts (sourced). PAIN uses its own environment (envs/pain.txt).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/PAIN}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
# determinism / thread / memory environment
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS=${SEEDS:-"1566911444 20241017 20251017"}
ARM=${ARM:?set ARM=original|universal|augmentation|invariant}
case "$ARM" in
  original) RUN_ARM=originals ;;          # the runner's internal name for the original arm
  universal|augmentation|invariant) RUN_ARM=$ARM ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
