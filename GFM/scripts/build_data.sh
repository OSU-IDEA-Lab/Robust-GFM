#!/bin/bash
# Builds every representation of WN18RR, FB15k-237 and Hetionet in the shared format (common/kg_format.py),
# then the compiled view of each one (common/compile_back.py). CPU only.
# Resources: ~64 GB RAM (FB15k-237 shortcut enumeration), ~15 min, ~0.6 GB of disk under GFM/data/.
# Needs: scripts/download_raw.sh (FB15k-237, Hetionet) and GNN/data/wordnet_3hops_augmented_full/ (WN18RR).
#
#   bash scripts/build_data.sh                  # all three datasets
#   bash scripts/build_data.sh FB15k237         # one dataset
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if (($#)); then DATASETS=("$@"); else DATASETS=(WN18RR FB15k237 Hetionet); fi

for DS in "${DATASETS[@]}"; do
  case "$DS" in
    WN18RR)
      python "$COMMON/build_wn18rr_variants.py" ;;
    FB15k237)
      python "$COMMON/build_kg_variants.py" --dataset FB15k237 --variants $(variants_of FB15k237) ;;
    Hetionet)
      # 5,000 test queries sampled by a keyed rule; the other held-out triples are used for filtering only
      python "$COMMON/build_kg_variants.py" --dataset Hetionet --variants $(variants_of Hetionet) \
          --max-queries 5000 --shortcut-max-paths 1000000 --reify-set regulation=CuG,CdG,DuG,DdG ;;
    *) echo "unknown dataset $DS" >&2; exit 2 ;;
  esac
  # compiled view: every <variant>__compiled must be byte-identical to <reference>__compiled
  python "$COMMON/compile_back.py" --dataset "$DS" --variants $(variants_of "$DS")
done
