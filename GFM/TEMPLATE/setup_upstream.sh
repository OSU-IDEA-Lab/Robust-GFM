#!/bin/bash
# Fetches the model's code at a fixed commit into upstream/ (never modified afterwards).
#
#   bash setup_upstream.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=https://github.com/<org>/<repo>.git        # set
COMMIT=<full commit hash>                        # set
CHECKPOINTS=()                                   # set: paths under upstream/ that must exist after checkout

if [[ ! -d "$HERE/upstream/.git" ]]; then
  git clone "$REPO" "$HERE/upstream"
fi
git -C "$HERE/upstream" fetch --quiet origin
git -C "$HERE/upstream" checkout --quiet "$COMMIT"
for f in "${CHECKPOINTS[@]}"; do
  [[ -f "$HERE/upstream/$f" ]] || { echo "missing upstream/$f" >&2; exit 1; }
done
echo "$(basename "$HERE") $COMMIT ready in $HERE/upstream"
