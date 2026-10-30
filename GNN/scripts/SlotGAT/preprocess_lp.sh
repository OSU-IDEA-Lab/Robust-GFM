#!/bin/bash
# SlotGAT has NO model-specific preprocessing for its LP cells (IMDb MY/MD, DBLP PC, WN18RR): the LP runners
# consume RGCN's structural bundles directly, so supervision, negatives and features are identical to RGCN/CMPNN/MAGNN.
# Build them with the RGCN preprocessing scripts (scripts/RGCN/preprocess_*.sh); this script checks they are present.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/SlotGAT}"
P=${INV_GNN_DATA:-data}/preprocessed
miss=0
for f in RGCN_structural_v10_year_canonfeat/IMDB/movie_year/{v1,v2,v3,v4,universal} \
         RGCN_structural_v10_year_canonfeat/IMDB/movie_director/{v1,v3,universal} \
         RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal} \
         RGCN_structural/WORDNET/universal_parity_full/full/intersection/{no_changes,all_inverse_edges,transitive_edges,universal_edges}; do
  if [ -f "$P/$f/graph.pt" ]; then echo "ok      $P/$f/graph.pt"; else echo "MISSING $P/$f/graph.pt"; miss=1; fi
done
exit $miss
