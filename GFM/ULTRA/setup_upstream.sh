#!/bin/bash
# Fetches ULTRA (MIT licence) at a fixed commit into ULTRA/upstream/. The released checkpoints
# (ultra_3g / ultra_4g / ultra_50g) are part of that repository, under upstream/ckpts/.
# The checkout is never modified; ultra_eval.py and ref_check.py apply their shims from outside it.
#
#   bash setup_upstream.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=https://github.com/DeepGraphLearning/ULTRA.git
COMMIT=427966ad8ed60420eef034063d44f3153addff90

if [[ ! -d "$HERE/upstream/.git" ]]; then
  git clone "$REPO" "$HERE/upstream"
fi
git -C "$HERE/upstream" fetch --quiet origin
git -C "$HERE/upstream" checkout --quiet "$COMMIT"
for ck in ultra_3g ultra_4g ultra_50g; do
  [[ -f "$HERE/upstream/ckpts/$ck.pth" ]] || { echo "missing upstream/ckpts/$ck.pth" >&2; exit 1; }
done
echo "ULTRA $COMMIT ready in $HERE/upstream"
