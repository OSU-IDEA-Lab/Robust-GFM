#!/usr/bin/env bash
# PAIN x Freebase: sampled rooted simple paths (L = 3; 4 relations / fan-out 8 / cap 256 per root) (CPU only).
# Input : data/dataset_variant_3hops_filtered/{unchanged,exact_2,exact_3,union_exact_2_3}/
# Output: data/preprocessed/pain_nc/Freebase/<variant>_L3_rel4_fan8_cap256.pt                 (~2 h, 6.9 GB)
#         data/preprocessed/pain_nc/Freebase_invariant/invariant_<variant>_L3_rel4_fan8_cap256.pt  (~3.7 h, 5.5 GB)
# The invariant step compiles every variant from its own base relations into the family program; the sampler is
# keyed on semantic identity, so the three path programs are byte-identical (status PASS in metadata.json).
# Resources: ~5 GB RAM. Both steps resume (--resume); rerun the script to continue after an interruption.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${MODEL_DIR:-$HERE/../../models/PAIN}"
if [ -n "${ENV_ACTIVATE:-}" ]; then source "$ENV_ACTIVATE"; fi
export OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
python -m preprocessing.freebase_node_classification --variants unchanged exact_2 exact_3 union_exact_2_3 --resume
python -m preprocessing.freebase_invariant --variants unchanged exact_2 exact_3 --resume
