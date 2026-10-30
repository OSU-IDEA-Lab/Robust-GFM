# Sourced by every GFM script. Plain bash; on a Slurm cluster wrap a script in sbatch/srun.
set -euo pipefail
GFM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON="$GFM_ROOT/common"
export GFM_DATA="${GFM_DATA:-$GFM_ROOT/data}"
# Optional environment activation (e.g. ENV_ACTIVATE=/path/to/venv/bin/activate).
if [[ -n "${ENV_ACTIVATE:-}" ]]; then source "$ENV_ACTIVATE"; fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONHASHSEED=0

# Representations of each dataset; the first entry is the reference every other one is compared with.
variants_of () {
  case "$1" in
    WN18RR)   echo "no_changes all_inverse_edges transitive_edges universal_edges" ;;
    FB15k237) echo "original inverse reified_cvt_strong reified_cvt_light shortcut2_c5k shortcut2_c50k shortcut2_c200k" ;;
    Hetionet) echo "original inverse reified_regulation shortcut2_c50k" ;;
    *) echo "unknown dataset $1 (WN18RR | FB15k237 | Hetionet)" >&2; return 2 ;;
  esac
}
reference_of () { local v; v=$(variants_of "$1"); echo "${v%% *}"; }
