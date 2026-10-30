# Shared setup for the PathNet scripts (sourced).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/PathNet}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
# determinism / thread environment
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SEEDS=${SEEDS:-"1566911444 20241017 20251017"}
ARM=${ARM:?set ARM=original|universal|augmentation|invariant}
case "$ARM" in
  original) RUN_ARM=originals ;;          # the runners' internal name for the original arm
  universal|augmentation|invariant) RUN_ARM=$ARM ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
