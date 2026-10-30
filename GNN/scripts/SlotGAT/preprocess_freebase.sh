#!/bin/bash
# SlotGAT Freebase NC inputs, built from the shared variant graphs
# data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3} (see dataset_construction/).
#   step 1  family context plans (one shared 129-channel catalog)  -> data/preprocessed/freebase_slotgat_block_family/<variant>     (CPU, 64 GB RAM, ~20 min, ~221 MB)
#   step 2  per-variant physical message schedules                   -> data/preprocessed/freebase_slotgat_physical_messages_family/{unchanged,exact_2,exact_3}
#           union message schedule (invariant + universal arms)       -> data/preprocessed/freebase_slotgat_union_messages/union_exact_2_3
#           + symlink freebase_slotgat_physical_messages_family/union_exact_2_3 -> the union schedule
#           (CPU, 96 GB RAM; exact_3 and union ~2.5 h and ~4.2 GB each; STEP=2 VARIANT=<v> builds one of them,
#           so the four can run as separate jobs)
# Usage: [STEP=1|2|all] [VARIANT=unchanged|exact_2|exact_3|union_exact_2_3] bash scripts/SlotGAT/preprocess_freebase.sh
set -euo pipefail
STEP=${STEP:-all}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export PYTHONHASHSEED=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
DATA=${INV_GNN_DATA:-data}
RAW=$DATA/dataset_variant_3hops_filtered
PLANS=$DATA/preprocessed/freebase_slotgat_block_family
PHYS=$DATA/preprocessed/freebase_slotgat_physical_messages_family
UNION=$DATA/preprocessed/freebase_slotgat_union_messages/union_exact_2_3
if [ "$STEP" = 1 ] || [ "$STEP" = all ]; then
  python preprocess_Freebase_block_conditional.py \
    --unchanged-dir "$RAW/unchanged" \
    --changed exact_2="$RAW/exact_2" --changed exact_3="$RAW/exact_3" --changed union_exact_2_3="$RAW/union_exact_2_3" \
    --registry registries/exact_2.jsonl --registry registries/exact_3.jsonl \
    --output-root "$PLANS" --overwrite
fi
if [ "$STEP" = 2 ] || [ "$STEP" = all ]; then
  for V in ${VARIANT:-unchanged exact_2 exact_3 union_exact_2_3}; do
    if [ "$V" = union_exact_2_3 ]; then OUT=$UNION; EXTRA=""; else OUT=$PHYS/$V; EXTRA="--any-context-kind"; fi
    python prepare_Freebase_union_message_equivalent.py \
      --exact2-dir "$RAW/$V" --context-plan-dir "$PLANS/$V" --output-dir "$OUT" $EXTRA --overwrite
    if [ "$V" = union_exact_2_3 ]; then
      mkdir -p "$PHYS"; ln -sfn ../freebase_slotgat_union_messages/union_exact_2_3 "$PHYS/union_exact_2_3"
    fi
  done
fi
echo "done (STEP=$STEP)"
