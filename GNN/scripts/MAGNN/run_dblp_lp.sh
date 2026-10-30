#!/bin/bash
# MAGNN x DBLP LP Paper-Conference (v1-v3 + universal).
#   ARM=original|universal|augmentation|invariant  [FLAVOUR=union|intersection]  [SEEDS="1566911444 20241017 20251017"]
# Prerequisite: scripts/MAGNN/preprocess_dblp.sh. Resources: one GPU (V100 32 GB), 4 CPU cores, 48 GB RAM.
# A run takes 1.5-3.5 h on a V100; to split the work, run one seed per call (SEEDS=...) and set e.g. RUN_TIMEOUT=7h
# (an interrupted run resumes per epoch when the script is called again).
# Example: ARM=invariant SEEDS=1566911444 scripts/MAGNN/run_dblp_lp.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DATASET=DBLP; TASK=paper_conference; VARIANTS="v1 v2 v3"
BUNDLES="$DATA/preprocessed/dblp_magnn_v2"
OUT="$RESULTS/dblp_lp/$ARM"
RA="$(runner_arm)"
gauntlet "$BUNDLES" "$VARIANTS" universal
source "$SCRIPT_DIR/_run_lp.sh"
