# Sourced by every ULTRA/scripts/*.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/_common.sh"
ULTRA="$GFM_ROOT/ULTRA"
RESULTS="${RESULTS:-$ULTRA/results}"
export RESULTS
[[ -d "$ULTRA/upstream/ultra" ]] || { echo "run ULTRA/setup_upstream.sh first" >&2; exit 2; }
# ULTRA's rspmm CUDA kernel is JIT-compiled on first use: it needs a C++17 compiler (CC / CXX) and nvcc.
# Build once for every GPU type you may run on, e.g. TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;9.0".
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ULTRA/torch_extensions}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ckpt_path () { echo "$ULTRA/upstream/ckpts/$1.pth"; }
