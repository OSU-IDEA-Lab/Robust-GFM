CUDA_MODULE_V100="${CUDA_MODULE_V100:-cuda/11.7}"

TORCH_EXTENSIONS_ROOT="${TORCH_EXTENSIONS_ROOT:-${HOME}/.cache/torch_extensions_cmpnn}"

# Make the `module` command available in non-interactive Slurm shells.
if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi

if ! type module >/dev/null 2>&1; then
    echo "ERROR: The environment-module command is unavailable."
    exit 1
fi

# PyTorch can identify the GPU assigned to the current Slurm job. Device 0 here
# means the first visible/allocated GPU, not necessarily physical GPU zero.
GPU_INFO="$(
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in the current job.")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
torch_cuda = torch.version.cuda or "unknown"
torch_version = torch.__version__.split("+", 1)[0]

print(f"{name}|{major}.{minor}|{torch_cuda}|{torch_version}")
PY
)"

IFS='|' read -r GPU_NAME GPU_CC TORCH_CUDA_VERSION TORCH_VERSION <<< "$GPU_INFO"

case "$GPU_CC" in
    7.0)
        GPU_KIND="v100"
        CUDA_MODULE="$CUDA_MODULE_V100"
        ;;

    *)
        echo "ERROR: Unsupported GPU compute capability: ${GPU_CC}"
        echo "Detected GPU: ${GPU_NAME}"
        echo "Add this GPU to the case statement in the launcher."
        exit 1
        ;;
esac

echo "Detected GPU: ${GPU_NAME}"
echo "Detected compute capability: ${GPU_CC}"
echo "PyTorch version: ${TORCH_VERSION}"
echo "PyTorch CUDA build: ${TORCH_CUDA_VERSION}"
echo "Loading CUDA module: ${CUDA_MODULE}"

# Remove a previously loaded CUDA module so it cannot leave a stale CUDA_HOME.
module unload cuda >/dev/null 2>&1 || true

if ! module load "$CUDA_MODULE"; then
    echo "ERROR: Failed to load CUDA module: ${CUDA_MODULE}"
    echo "Run 'module avail cuda' and update CUDA_MODULE_V100."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc is unavailable after loading ${CUDA_MODULE}."
    exit 1
fi

NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
export CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

if command -v gcc >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
fi

if command -v g++ >/dev/null 2>&1; then
    export CXX="$(command -v g++)"
fi

# Compile specifically for the GPU assigned to this job:
#   V100 -> 7.0
export TORCH_CUDA_ARCH_LIST="${GPU_CC}"

NVCC_VERSION="$(
nvcc --version |
    sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' |
    tail -n 1
)"

if [[ -z "$NVCC_VERSION" ]]; then
    NVCC_VERSION="unknown"
fi

# Keep separate caches by GPU architecture, PyTorch version, PyTorch CUDA build,
# and nvcc version, so an extension built for another GPU is never loaded.
SAFE_TORCH_VERSION="${TORCH_VERSION//./_}"
SAFE_TORCH_CUDA="${TORCH_CUDA_VERSION//./_}"
SAFE_NVCC_VERSION="${NVCC_VERSION//./_}"

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_ROOT}/${GPU_KIND}_sm${GPU_CC//./}_torch${SAFE_TORCH_VERSION}_cu${SAFE_TORCH_CUDA}_nvcc${SAFE_NVCC_VERSION}"

mkdir -p "$TORCH_EXTENSIONS_DIR"

# Limit parallel compiler processes. This affects only initial JIT compilation.
export MAX_JOBS="${MAX_JOBS:-4}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDACXX=${CUDACXX}"
echo "nvcc version=${NVCC_VERSION}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "MAX_JOBS=${MAX_JOBS}"

# Verify that PyTorch still sees the expected GPU.
python - <<'PY'
import os
import torch

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)

print("CUDA configuration check passed")
print(f"  GPU: {name}")
print(f"  Compute capability: {major}.{minor}")
print(f"  PyTorch CUDA build: {torch.version.cuda}")
print(f"  TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")
print(f"  TORCH_EXTENSIONS_DIR: {os.environ['TORCH_EXTENSIONS_DIR']}")
PY