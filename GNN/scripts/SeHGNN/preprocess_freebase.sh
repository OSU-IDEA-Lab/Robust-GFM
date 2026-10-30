#!/bin/bash
# SeHGNN x Freebase: compile the relation-aware full-K (`full_rel`) channel bundles read by the four arms.
#
#   prerequisite : data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/
#                  {info.dat,node.dat,link.dat,label.dat[,new_transitive_relations.jsonl]}   (dataset_construction/)
#   output       : data/preprocessed/sehgnn_v2/FREEBASE/full_rel/semantic/{unchanged,exact_2,exact_3}/     (invariant;
#                      the 597-channel semantic program -- the invariant run keeps its 511 channels that occur in a
#                      variant's own stock bias, read from the physical bundles below)
#                  data/preprocessed/sehgnn_v2/FREEBASE/full_rel/physical/{unchanged,exact_2,exact_3}/     (original,
#                      augmentation; reference for the invariant channel subset)
#                  data/preprocessed/sehgnn_v2/FREEBASE/full_rel/physical/union_exact_2_3/                (universal)
# CPU only (8 threads). Each variant takes ~10-40 min (loading the 0.9-9.3 GB link.dat dominates) and up to ~128 GB RAM
# (exact_3 / union_exact_2_3). Restrict with VARIANTS= / GRAPHS= to build one variant per call, e.g.
#   VARIANTS=exact_3 bash scripts/SeHGNN/preprocess_freebase.sh
set -euo pipefail
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
${ENV_ACTIVATE:-}
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONHASHSEED=0
export INV_GNN_DATA="${INV_GNN_DATA:-$REL/data}"
cd "$REL/models/SeHGNN"
OUT="$INV_GNN_DATA/preprocessed/sehgnn_v2"
VARIANTS="${VARIANTS:-unchanged exact_2 exact_3 union_exact_2_3}"
GRAPHS="${GRAPHS:-semantic physical}"
for V in $VARIANTS; do
  for G in $GRAPHS; do
    [ "$V" = union_exact_2_3 ] && [ "$G" = semantic ] && continue        # union graph = universal arm, physical only
    echo "===== $V $G full_rel $(date -Is) ====="
    python preprocess_sehgnn_channels.py --dataset FREEBASE --variant "$V" --graph "$G" --flavor full_rel --out-root "$OUT"
  done
done
if [ "${GAUNTLET:-1}" = 1 ] && [ -d "$OUT/FREEBASE/full_rel/semantic/exact_3" ]; then
  python validate_sehgnn_invariance.py --dataset FREEBASE --flavors full_rel --data-root "$OUT" --skip-first-step \
    --out "$OUT/FREEBASE/gauntlet.json" || echo "[gauntlet] FAILED or incomplete bundle set"
fi
