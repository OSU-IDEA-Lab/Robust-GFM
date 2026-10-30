#!/bin/bash
# Collects every comparison under $RESULTS into $RESULTS/summary.{csv,md} and draws $RESULTS/figure.{pdf,png}.
# CPU only, seconds.
#
#   bash scripts/summarize.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OPENBLAS_NUM_THREADS=1 python "$ULTRA/aggregate.py"
python "$ULTRA/make_figure.py"
