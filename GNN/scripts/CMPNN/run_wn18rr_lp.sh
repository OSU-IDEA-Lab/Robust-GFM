#!/bin/bash
# CMPNN link prediction -- WN18RR (WORDNET / link_prediction), one arm per call.
#   ARM=original      run_cmpnn_lp.py --graph physical, variants no_changes, all_inverse_edges, transitive_edges (= WN18RR1-3)
#   ARM=universal     run_cmpnn_lp.py --graph physical --variant universal_edges   (CMPNN* on the union graph)
#   ARM=augmentation  run_cmpnn_lp_augmentation.py, one model trained jointly on the three physical graphs
#   ARM=invariant     run_cmpnn_lp_wordnet_context.py --graph semantic, every variant: the semantic program restricted
#                     to the 11 base:: relations (inverses / 3-hop shortcuts are context only)
#   ARM=gauntlet      verification step, not an experiment: data-level checks of the bundles (validate_cmpnn_invariance.py), no training
# SEEDS (default "1566911444 20241017 20251017"), INV_GNN_DATA (default <release>/data), RESULTS_ROOT,
# ENV_ACTIVATE (optional), CMPNN_MODULE_SETUP=1 (optional CUDA-module setup).
# EXTRA default "--batch-size 64 --chunk-edges 200000" (fits a V100 32 GB).
# Prerequisite: scripts/RGCN/preprocess_wn18rr.sh (bundle RGCN_structural/WORDNET/universal_parity_full).
# Contract (run_cmpnn_lp.py DEFAULTS): 6 layers, 32/32, SUM aggregation, DistMult, lr 5e-3, 32 training negatives,
# <=10 epochs, patience 3 (val filtered MRR), remove_one_hop=False; filtered full-entity ranking.
# Resources: 2-17 h and ~17 GiB per run on a V100 (see models/CMPNN/README.md); runs are resumable per epoch (--resume, re-run the same command).
EXTRA="${EXTRA---batch-size 64 --chunk-edges 200000}"
source "$(dirname "${BASH_SOURCE[0]}")/_cmpnn_common.sh"
DS=WORDNET; TASK=link_prediction; VARIANTS="no_changes,all_inverse_edges,transitive_edges"
OUT="$RESULTS_ROOT/WN18RR_link_prediction/$ARM"
echo "host=$(hostname) DS=$DS TASK=$TASK ARM=$ARM SEEDS=$SEEDS OUT=$OUT EXTRA=$EXTRA start=$(date -Is)"
case "$ARM" in
  gauntlet) cmpnn_gauntlet "$DS" "$TASK" "$VARIANTS,universal_edges" || { echo "GAUNTLET FAILED"; exit 9; }; exit 0 ;;
esac
for SD in $SEEDS; do
  case "$ARM" in
    original)     for V in ${VARIANTS//,/ }; do cmpnn_run_one "$DS" "$TASK" "$OUT" physical "$V" "$SD"; done ;;
    universal)    cmpnn_run_one "$DS" "$TASK" "$OUT" physical universal_edges "$SD" ;;
    invariant)    for V in ${VARIANTS//,/ }; do cmpnn_run_one "$DS" "$TASK" "$OUT" semantic "$V" "$SD" run_cmpnn_lp_wordnet_context.py; done ;;
    augmentation) cmpnn_run_aug "$DS" "$TASK" "$OUT" "$VARIANTS" "$SD" "" ;;
    *) echo "unknown ARM=$ARM"; exit 2 ;;
  esac
done
cmpnn_finish "$OUT"
