#!/bin/bash
# SeHGNN x Freebase node classification ({unchanged, exact_2, exact_3} = Freebase1-3; union graph union_exact_2_3),
# one arm x seeds, relation-aware full-K channels (every semantic relation its own step, K=2, --batch-size 1024).
#   original / universal / augmentation  channel set full_rel
#   invariant                            channel set full_rel_uov511
#                                        (511 of the 597 semantic channels: those occurring in at least one variant's
#                                        own stock bias; run_sehgnn_nc_subset.py --subset union_of_variants)
#
#   ARM      = original | universal | augmentation | invariant                                   (required)
#   SEEDS    = "1566911444 20241017 20251017" (default)
#   OUT      = output root (default models/SeHGNN/results) ->
#              $OUT/FREEBASE_nc/<armdir>_<channels>/FREEBASE/nc/<variant>/seed_<s>/
#              (<armdir>_<channels> = originals_full_rel | universal_full_rel | augmentation_full_rel | invariant_full_rel_uov511)
#   AGGREGATE=1 also writes .../aggregate/ via common/aggregate_rgcn_structural.py
#   EXTRA    = extra runner flags (none needed for the documented results)
# Needs: bash scripts/SeHGNN/preprocess_freebase.sh; torch_sparse (PyG extra) for the sparse channel selectors
# (the bitwise-deterministic backend).
# Resources: one GPU (V100 32 GB), 4 CPU cores, 64 GB RAM; ~3-6 min per run, peak GPU memory <= 17 GB.
# Example: ARM=invariant bash scripts/SeHGNN/run_freebase_nc.sh
set -uo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
${ENV_ACTIVATE:-}
export DGLBACKEND=pytorch OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export INV_GNN_DATA="${INV_GNN_DATA:-$REL/data}"
cd "$REL/models/SeHGNN"
ARM="${ARM:?set ARM=original|universal|augmentation|invariant}"
SEEDS="${SEEDS:-1566911444 20241017 20251017}"
OUT="${OUT:-results}"
EXTRA="${EXTRA:-}"
DATA="$INV_GNN_DATA/preprocessed/sehgnn_v2"
VARIANTS="unchanged exact_2 exact_3"
BS="--batch-size 1024"      # the 511/597-channel attention does not fit full batch on a 32 GB GPU
FAILED=0
for SD in $SEEDS; do
  case "$ARM" in
    invariant)
      for V in $VARIANTS; do
        python run_sehgnn_nc_subset.py --dataset FREEBASE --arm invariant --flavor full_rel --variant "$V" --seed "$SD" --data-root "$DATA" \
          --output-root "$OUT" --resume $BS --subset union_of_variants --reference-root "$DATA/FREEBASE/full_rel/physical" --tag _uov511 $EXTRA || FAILED=1
      done ;;
    original)
      for V in $VARIANTS; do
        python run_sehgnn_nc.py --dataset FREEBASE --arm originals --flavor full_rel --variant "$V" --seed "$SD" --data-root "$DATA" \
          --output-root "$OUT" --resume $BS $EXTRA || FAILED=1
      done ;;
    universal)
      python run_sehgnn_nc.py --dataset FREEBASE --arm universal --flavor full_rel --variant union_exact_2_3 --seed "$SD" --data-root "$DATA" \
        --output-root "$OUT" --resume $BS $EXTRA || FAILED=1 ;;
    augmentation)
      python run_sehgnn_nc_augmentation.py --dataset FREEBASE --flavor full_rel --variants unchanged,exact_2,exact_3 --seed "$SD" --data-root "$DATA" \
        --output-root "$OUT" --resume $BS $EXTRA || FAILED=1 ;;
    *) echo "unknown ARM=$ARM"; exit 2 ;;
  esac
done
if [ "${AGGREGATE:-0}" = 1 ]; then
  case "$ARM" in original) D=originals_full_rel;; invariant) D=invariant_full_rel_uov511;; *) D="${ARM}_full_rel";; esac
  D="$OUT/FREEBASE_nc/$D"
  python ../../common/aggregate_rgcn_structural.py --results-root "$D" --out-dir "$D/aggregate" | tail -3
fi
exit $FAILED
