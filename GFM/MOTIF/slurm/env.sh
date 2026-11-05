# Shared job environment (docs/HPC_ZERO_SHOT_RUNBOOK.md), sourced by the sbatch scripts.
module load python/3.10 gcc/12.2 cuda/12.1
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export CC="$(command -v gcc)" CXX="$(command -v g++)"
# One extension cache per GPU architecture (7.0 V100, 7.5 RTX 2080/8000, 8.6 A40, 9.0 H100/H200).
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export TORCH_CUDA_ARCH_LIST="$ARCH"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions_cu121_sm${ARCH/./}"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export MAX_JOBS=4 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
