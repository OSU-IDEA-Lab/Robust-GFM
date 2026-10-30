# Sourced by every scripts/RGCN/*.sh. Plain bash; on a Slurm cluster wrap it in sbatch/srun.
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RGCN="$RELEASE_ROOT/models/RGCN"
AUG="$RGCN/augmentation"
# Optional environment activation (e.g. ENV_ACTIVATE=/path/to/venv/bin/activate).
if [[ -n "${ENV_ACTIVATE:-}" ]]; then source "$ENV_ACTIVATE"; fi
# Thread settings used for the expected results. The runners themselves set
# CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED / torch deterministic mode internally.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export DGLBACKEND=pytorch
# Data: models/RGCN/data -> ../../data (= $RELEASE_ROOT/data); all code paths are data/... relative to the model dir.
DATA="$RGCN/data"
RESULTS="${RESULTS:-$RGCN/results}"
ARM="${ARM:-}"
SEEDS="${SEEDS:-1566911444 20241017 20251017}"
DEVICE="${DEVICE:-cuda:0}"
require_arm () {
  case "$ARM" in original|universal|augmentation|invariant) ;; *)
    echo "set ARM=original|universal|augmentation|invariant" >&2; exit 2;; esac
}
comma () { echo "$*" | tr ' ' ','; }
