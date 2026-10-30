#!/bin/bash
# CMPNN link prediction -- DBLP / paper_conference, one arm per call.
#   ARM=original      run_cmpnn_lp.py --graph physical, every variant (v1,v2,v3)
#   ARM=universal     run_cmpnn_lp.py --graph physical --variant universal   (CMPNN* on the union graph)
#   ARM=augmentation  run_cmpnn_lp_augmentation.py, one model trained jointly on all variants' physical graphs
#   ARM=invariant     run_cmpnn_lp.py --graph semantic, every variant (compiled semantic program = ours)
#   ARM=gauntlet      verification step, not an experiment: data-level checks of the bundles (validate_cmpnn_invariance.py), no training
# SEEDS (default "1566911444 20241017 20251017"), INV_GNN_DATA (default <release>/data), RESULTS_ROOT,
# ENV_ACTIVATE (optional), CMPNN_MODULE_SETUP=1 (optional CUDA-module setup), EXTRA (extra runner flags; empty = default config).
# Prerequisite: scripts/RGCN/preprocess_dblp.sh (bundle RGCN_structural_v8_dblp).
# Resources: one V100 32 GB; ~4.8 h and ~11 GiB per run, so run one arm or seed per job.
# Contract (run_cmpnn_lp.py DEFAULTS): 6 layers, 32/32, PNA, DistMult, lr 1e-3, batch 8, <=20 epochs, patience 5 (val MRR), remove_one_hop.
# Output: models/CMPNN/results/DBLP_paper_conference/<arm>/DBLP/paper_conference/<variant>/seed_<s>/{metrics.json,scores.npz,checkpoint.pt,trace.json}
source "$(dirname "${BASH_SOURCE[0]}")/_cmpnn_common.sh"
DS=DBLP; TASK=paper_conference; VARIANTS="v1,v2,v3"
OUT="$RESULTS_ROOT/${DS}_${TASK}/$ARM"
echo "host=$(hostname) DS=$DS TASK=$TASK ARM=$ARM SEEDS=$SEEDS OUT=$OUT EXTRA=$EXTRA start=$(date -Is)"
case "$ARM" in
  gauntlet) cmpnn_gauntlet "$DS" "$TASK" "$VARIANTS,universal" || { echo "GAUNTLET FAILED"; exit 9; }; exit 0 ;;
esac
for SD in $SEEDS; do
  case "$ARM" in
    original)     for V in ${VARIANTS//,/ }; do cmpnn_run_one "$DS" "$TASK" "$OUT" physical "$V" "$SD"; done ;;
    universal)    cmpnn_run_one "$DS" "$TASK" "$OUT" physical universal "$SD" ;;
    invariant)    for V in ${VARIANTS//,/ }; do cmpnn_run_one "$DS" "$TASK" "$OUT" semantic "$V" "$SD"; done ;;
    augmentation) cmpnn_run_aug "$DS" "$TASK" "$OUT" "$VARIANTS" "$SD" "--resume" ;;
    *) echo "unknown ARM=$ARM"; exit 2 ;;
  esac
done
cmpnn_finish "$OUT"
