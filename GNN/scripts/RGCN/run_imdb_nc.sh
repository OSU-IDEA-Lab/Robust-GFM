#!/usr/bin/env bash
# RGCN, IMDb (year graph, canonical pooled-BoW features), task nc.
# Usage: ARM=original|universal|augmentation|invariant [SEEDS="..."] [DEVICE=cuda:0] bash scripts/RGCN/run_imdb_nc.sh
TASK=nc
source "$(dirname "${BASH_SOURCE[0]}")/_imdb.sh"
