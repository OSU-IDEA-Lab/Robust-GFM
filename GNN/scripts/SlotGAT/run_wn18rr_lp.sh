#!/bin/bash
# SlotGAT x WN18RR link prediction (models/SlotGAT/lp). Flag on top of the runner defaults: --loss neg_logsigmoid
# (defaults: hidden 128, 2x4 heads, concat_ln readout, batch 2048 x 256 negatives, lr 1e-3, eval every 5, patience 8;
# each run also saves its full configuration in metrics.json).
#   ARM=original     -> --arm originals   (no_changes, all_inverse_edges, transitive_edges; physical graphs)
#   ARM=universal    -> --arm universal   (universal_edges = union, physical typing)
#   ARM=augmentation -> --arm augmentation
#   ARM=invariant    -> --arm invariant   (each variant on the shared semantic program)
# Results: models/SlotGAT/results/wn18rr_lp/<originals|...>/full/intersection/<variant>/seed_<s>/
#          aggregate: python lp/aggregate_lp_arms.py --family wordnet_nls
# Inputs: data/preprocessed/RGCN_structural/WORDNET/universal_parity_full/full/intersection/<variant>/graph.pt (RGCN builder).
# Cost: ~17 s/epoch on a V100, one run < 2 h. Resources: one GPU, 2 CPUs, 48 GB RAM.
# Usage: ARM=invariant [SEEDS="..."] bash scripts/SlotGAT/run_wn18rr_lp.sh
set -euo pipefail
ARM=${ARM:?set ARM=original|universal|augmentation|invariant}
SEEDS=${SEEDS:-"1566911444 20241017 20251017"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=${OUT_ROOT:-results/wn18rr_lp}
EXTRA="--loss neg_logsigmoid"
case "$ARM" in original) A=originals ;; universal|augmentation|invariant) A=$ARM ;; *) echo "unknown ARM=$ARM" >&2; exit 2 ;; esac
for SEED in $SEEDS; do
  R="python lp/run_wordnet_lp_arms.py --out-root $OUT --seed $SEED $EXTRA"
  case $A in
    originals|invariant) for V in no_changes all_inverse_edges transitive_edges; do $R --arm $A --variant $V; done ;;
    *) $R --arm $A ;;
  esac
done
echo "done: ARM=$ARM -> $(pwd)/$OUT/$A"
