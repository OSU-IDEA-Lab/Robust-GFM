#!/usr/bin/env bash
# RGCN Freebase stores (CPU only; needs ~48 GB RAM and lots of disk -- see models/RGCN/README.md).
# Input : data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/{node,link,label,info}.dat
#         (built by dataset_construction/), catalogs in models/RGCN/catalogs/ (shipped).
# Output: data/preprocessed/freebase_rgcn_physical/<V>/physical_<V>/union/all_graph            (original / universal / augmentation)
#         data/preprocessed/freebase_rgcn_block_conditional/<V>/exact_3/union/all_graph          (invariant)
# STEPS="physical invariant" by default (physical = original/universal/augmentation stores,
# invariant = exact_3 block-conditional stores).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export PYTHONPATH="$RGCN${PYTHONPATH:+:$PYTHONPATH}"
cd "$RGCN"
VROOT=data/dataset_variant_3hops_filtered
STEPS="${STEPS:-physical invariant}"

compile () {  # $1 variant  $2 reference variant  $3 catalog (--up-to)  $4 output dir
  if [[ -f "$4/model_input_hashes.json" ]]; then echo "[$1/$3] already compiled"; return; fi
  python preprocess_Freebase_rgcn_block_conditional.py --package-root "$RGCN" \
    --data-dir "$VROOT/$1" --reference-dir "$VROOT/$2" --output-dir "$4" \
    --input-variant "$1" --up-to "$3" --set-mode union --scope all_graph --verify-physical counts --overwrite
}

if [[ " $STEPS " == *" physical "* ]]; then
  PRE=data/preprocessed/freebase_rgcn_physical
  for V in unchanged exact_2 exact_3 union_exact_2_3; do
    # regenerates models/RGCN/catalogs/physical_<V>/union/all_graph.json (shipped; idempotent)
    python build_physical_only_catalog.py --variant-dir "$VROOT/$V" --name "physical_$V"
    # reference = the variant itself (its info.dat knows the materialised shortcut relations)
    compile "$V" "$V" "physical_$V" "$PRE/$V/physical_$V/union/all_graph"
  done
fi

block_family () {  # $1 catalog
  PRE=data/preprocessed/freebase_rgcn_block_conditional
  REF=$PRE/unchanged/$1/union/all_graph
  for V in unchanged exact_2 exact_3; do
    compile "$V" unchanged "$1" "$PRE/$V/$1/union/all_graph"
    # optional disk saving: hard-link channel files identical to the reference (identical by construction)
    [[ "$V" == unchanged ]] || python dedupe_freebase_channels.py --reference "$REF" --target "$PRE/$V/$1/union/all_graph"
  done
  # cross-variant pretraining invariance audit (identical sampled blocks / model inputs)
  mkdir -p "$RESULTS/freebase_nc/audit_$1"
  python validate_Freebase_rgcn_block_invariance.py "$REF" "$PRE/exact_2/$1/union/all_graph" "$PRE/exact_3/$1/union/all_graph" \
    --fanout-layer1 4 --fanout-layer2 4 --output "$RESULTS/freebase_nc/audit_$1/pretraining_invariance_audit.json"
}
[[ " $STEPS " == *" invariant "* ]] && block_family exact_3
echo "[done] Freebase stores"
