#!/bin/bash
# Clones SCR at a fixed commit into SCR/upstream/. That checkout is the model
# code (package ultra, configs). This folder's runners are not copied into it.
#
# The released checkpoint ckpts/scr_3g.pth and the resource/ text files ship
# with that repository when they are present at this commit. resource/ is large
# and may need the Google Drive bundle named in the SCR README.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=https://github.com/KyneWang/SCR.git
COMMIT=5e46b54296b9e6b30d83cb2f3779da3376a3fcb8

if [[ ! -d "$HERE/upstream/.git" ]]; then
  git clone "$REPO" "$HERE/upstream"
fi
git -C "$HERE/upstream" fetch --quiet origin
git -C "$HERE/upstream" checkout --quiet "$COMMIT"
[[ -f "$HERE/upstream/ckpts/scr_3g.pth" ]] || {
  echo "missing upstream/ckpts/scr_3g.pth" >&2
  exit 1
}
echo "SCR $COMMIT ready in $HERE/upstream"
