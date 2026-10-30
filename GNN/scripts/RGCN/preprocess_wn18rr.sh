#!/usr/bin/env bash
# RGCN WN18RR bundle -- SHARED PREREQUISITE (also read by CMPNN and SlotGAT (LP)).
# Input : data/wordnet_3hops_augmented_full/wordnet_splits.npz  (dataset_construction/; holds every variant's
#         train edges + the official val/test triples), models/RGCN/wordnet_structural_catalogs/full_selection.json (shipped)
# Output: data/preprocessed/RGCN_structural/WORDNET/universal_parity_full/full/intersection/
#         {no_changes,all_inverse_edges,transitive_edges,universal_edges}/{graph.pt,audit.json} + preprocess_summary.json (~410 MB)
# Expected semantic hash 64d2c91fa9d7446d2978ce0d6760c6359f416dd920de56badf58eddf5864745a (preprocess_summary.json).
# CPU only, ~4 min, ~24 GB RAM.
# Set VALIDATE=1 to also run the pre-training invariance audit (validate_WORDNET_rgcn_structural.py, CPU).
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RGCN"
PYTHON_BIN=python DEVICE="$DEVICE" \
VARIANTS="no_changes all_inverse_edges transitive_edges universal_edges" \
BUDGETS_TO_RUN=full MODES_TO_RUN=intersection \
DATASET_ROOT="$RGCN/data/wordnet_3hops_augmented_full" \
PREPROCESSED_ROOT="$RGCN/data/preprocessed/RGCN_structural/WORDNET/universal_parity_full" \
RESULTS_ROOT="$RESULTS/wn18rr_lp/preprocess_audit" \
MAX_ENDPOINTS_PER_SOURCE=0 OVERWRITE_PREPROCESS="${OVERWRITE_PREPROCESS:-0}" \
RUN_TESTS=0 PREPARE_SPLITS=0 PREPROCESS=1 VALIDATE="${VALIDATE:-0}" RUN_PARITY_DIAGNOSTIC=0 TRAIN=0 AGGREGATE=0 \
bash run_wordnet_rgcn_structural_full.sh
