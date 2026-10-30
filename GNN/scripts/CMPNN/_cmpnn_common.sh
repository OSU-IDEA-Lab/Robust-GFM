#!/bin/bash
# Shared helper for scripts/CMPNN/run_*.sh -- sourced, not executed.
# Runs are skip-if-done (metrics.json) and use --resume on the per-variant runs.
set -uo pipefail

RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMPNN="$RELEASE/models/CMPNN"
INV_GNN_DATA="${INV_GNN_DATA:-$RELEASE/data}"
DATA_ROOT="$INV_GNN_DATA/preprocessed"
SEEDS="${SEEDS:-1566911444 20241017 20251017}"
ARM="${ARM:?set ARM=original|universal|augmentation|invariant (ARM=gauntlet = data-level verification only)}"
EXTRA="${EXTRA:-}"

# optional environment activation (e.g. ENV_ACTIVATE="source /path/to/conda/bin/activate torchdrug_env")
if [ -n "${ENV_ACTIVATE:-}" ]; then eval "$ENV_ACTIVATE"; fi
# thread settings used for the expected results
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd "$CMPNN"
# optional: CUDA-module / torchdrug JIT-extension-cache setup (needs `module`; V100 -> cuda/11.7)
if [ "${CMPNN_MODULE_SETUP:-0}" = "1" ]; then source scripts/setup.sh || exit 1; fi

RESULTS_ROOT="${RESULTS_ROOT:-$CMPNN/results}"
FILTER='Warning\|pkg_resources\|bincount\|F.linear\|run_backward'
ANY_FAILED=0

# $1=dataset $2=task $3=arm-dir $4=graph(semantic|physical) $5=variant $6=seed [$7=runner]
cmpnn_run_one () {
  local ds="$1" task="$2" out="$3" graph="$4" v="$5" sd="$6" runner="${7:-run_cmpnn_lp.py}"
  local dir="$out/$ds/$task/$v/seed_$sd"
  if [ -f "$dir/metrics.json" ]; then echo "[skip] $dir"; return 0; fi
  echo "[start] $runner $ds/$task $graph $v seed_$sd $(date -Is)"
  PYTHONPATH=. python "$runner" --data-root "$DATA_ROOT" --dataset "$ds" --task "$task" --variant "$v" --seed "$sd" \
    --graph "$graph" --output-root "$out" --resume $EXTRA 2>&1 | grep --line-buffered -v "$FILTER"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "[FAILED rc=$rc] $ds/$task $v seed_$sd"; ANY_FAILED=1; }
  return 0
}

# $1=dataset $2=task $3=arm-dir $4=variants(comma) $5=seed $6=extra flags for the augmentation runner
cmpnn_run_aug () {
  local ds="$1" task="$2" out="$3" vs="$4" sd="$5" aflags="${6:-}"
  if ls "$out/$ds/$task"/*/seed_$sd/metrics.json >/dev/null 2>&1; then echo "[skip] augmentation seed_$sd"; return 0; fi
  echo "[start] augmentation $ds/$task seed_$sd $(date -Is)"
  PYTHONPATH=. python run_cmpnn_lp_augmentation.py --data-root "$DATA_ROOT" --dataset "$ds" --task "$task" --variants "$vs" \
    --seed "$sd" --output-root "$out" $aflags $EXTRA 2>&1 | grep --line-buffered -v "$FILTER"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "[FAILED rc=$rc] augmentation seed_$sd"; ANY_FAILED=1; }
  return 0
}

# data-level gauntlet (hashes, duplicate edges, <=2-hop leak audit); run it before training
cmpnn_gauntlet () {   # $1=dataset $2=task $3=variants incl. universal (comma)
  mkdir -p "$RESULTS_ROOT/${1}_${2}"
  PYTHONPATH=. python validate_cmpnn_invariance.py --data-root "$DATA_ROOT" --dataset "$1" --task "$2" \
    --variants "$3" --report "$RESULTS_ROOT/${1}_${2}/gauntlet_data_level.json" 2>&1 | grep -v 'Warning\|pkg_resources'
  return "${PIPESTATUS[0]}"
}

cmpnn_finish () {   # $1 = arm output dir: per-arm aggregate (per-variant means, pairwise Kendall tau, checkpoint-hash equality)
  # with the RGCN aggregator (AGGREGATE=0 skips it)
  if [ "${AGGREGATE:-1}" = "1" ] && [ -n "${1:-}" ] && [ -d "$1" ] && [ -f "$RELEASE/common/aggregate_rgcn_structural.py" ]; then
    python "$RELEASE/common/aggregate_rgcn_structural.py" --results-root "$1" --out-dir "$1/aggregate" 2>&1 | tail -1
  fi
  [ "$ANY_FAILED" -ne 0 ] && { echo "===== FAILED $(date -Is) ====="; exit 1; }
  echo "===== done $(date -Is) ====="
}
