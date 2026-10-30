#!/usr/bin/env bash
# RGCN DBLP inputs. CPU only, ~2 minutes.
# Input : data/raw/DBLP/{author_label,paper_author,paper_conf,paper_term}.txt
#         data/preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz   (dataset_construction/)
# Output: 1) data/preprocessed/RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}  (~110 MB)
#            -- SHARED PREREQUISITE: also read by CMPNN, MAGNN (DBLP) and SlotGAT (LP).
#            Expected semantic hash 308cec36b62b on every variant.
#         2) data/preprocessed/DBLP_rgcn_{v1,v2,v3}_canonical  (graph_data.pt + meta.pt; augmentation arm only)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$RGCN"
python preprocess_DBLP_rgcn_structural.py \
  --raw-dir data/raw/DBLP \
  --shared-npz data/preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz \
  --variants v1,v2,v3,universal \
  --out-root data/preprocessed/RGCN_structural_v8_dblp
python preprocess_DBLP_rgcn.py --variant v1,v2,v3 \
  --raw-dir data/raw/DBLP/ \
  --shared-npz data/preprocessed/DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz \
  --out-dir data/preprocessed --data-suffix _canonical
