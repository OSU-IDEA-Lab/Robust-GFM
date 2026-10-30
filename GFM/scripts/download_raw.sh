#!/bin/bash
# Downloads the raw FB15k-237 and Hetionet triples into GFM/data/<DATASET>/raw/ and checks their sha256.
# WN18RR is not downloaded here: it is exported from GNN/data/wordnet_3hops_augmented_full/ (see GNN/data/README.md).
# Resources: network access, < 200 MB of disk.
#
#   bash scripts/download_raw.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Same files as PyG's FB15k-237 and ULTRA's Hetionet split.
FB=https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237
declare -A URL=(
  [FB15k237/train.txt]="$FB/train.txt"
  [FB15k237/valid.txt]="$FB/valid.txt"
  [FB15k237/test.txt]="$FB/test.txt"
  [Hetionet/train.txt]="https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1"
  [Hetionet/valid.txt]="https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1"
  [Hetionet/test.txt]="https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1"
)
declare -A SHA=(
  [FB15k237/train.txt]=61099230e4439f90885ca9767739e31e8e32f54736fa1c35952b27997bc7c08a
  [FB15k237/valid.txt]=749cbe9d923bac7b9354da5614ecfed2e0220256d442c3e04a6b303db1f273d9
  [FB15k237/test.txt]=e2e35e8e6113de220140b6f44dc71a5207b0fc6872d575e874aefe13259b655b
  [Hetionet/train.txt]=cc8edc84e9e62a955affc88e21eeba7f272c04d552e53e780624c5b75ad929b2
  [Hetionet/valid.txt]=7c8301fb0e3635e347812111afd02580c2dd6c915dea5c4641d4aff2108f36ed
  [Hetionet/test.txt]=53afde5426cb915241887e9a7e83ba979a6bfb43a433cc25e8b552c7ec768728
)
# Set CURL_CA_BUNDLE if curl cannot find the system CA certificates.
CURL=(curl -fsSL --retry 3)
[[ -n "${CURL_CA_BUNDLE:-}" ]] && CURL+=(--cacert "$CURL_CA_BUNDLE")

for key in "${!URL[@]}"; do
  ds=${key%%/*}; file=${key#*/}
  out="$GFM_DATA/$ds/raw/$file"
  mkdir -p "$(dirname "$out")"
  if [[ ! -f "$out" ]]; then
    echo "download $key"
    "${CURL[@]}" -o "$out.part" "${URL[$key]}"
    mv "$out.part" "$out"
  fi
  got=$(sha256sum "$out" | cut -d' ' -f1)
  [[ "$got" == "${SHA[$key]}" ]] || { echo "sha256 mismatch for $out" >&2; exit 1; }
done
echo "raw triples OK under $GFM_DATA"
