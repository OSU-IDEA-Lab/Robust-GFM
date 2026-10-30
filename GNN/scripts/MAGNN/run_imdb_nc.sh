#!/bin/bash
# MAGNN x IMDb node classification (year graph, v1-v4 + universal).
#   ARM=original|universal|augmentation|invariant  [FLAVOUR=union|intersection]  [SEEDS="1566911444 20241017 20251017"]
# Prerequisite: scripts/MAGNN/preprocess_imdb.sh. Resources: one GPU (V100 32 GB), 4 CPU cores, 48 GB RAM; ~2-5 min per run.
# Example: ARM=invariant FLAVOUR=union scripts/MAGNN/run_imdb_nc.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BUNDLES="$DATA/preprocessed/imdb_magnn_v2/nc"
VARIANTS="v1 v2 v3 v4"
OUT="$RESULTS/imdb_nc/$ARM"
RA="$(runner_arm)"
gauntlet "$BUNDLES" "$VARIANTS" universal
for SD in $SEEDS; do
  case "$RA" in
    invariant|invariant_union|originals) VS="$VARIANTS" ;;
    universal) VS="universal" ;;
    augmentation)
      if ls "$OUT/IMDB/nc"/*/seed_$SD/metrics.json >/dev/null 2>&1; then echo "[skip] augmentation seed_$SD"; continue; fi
      _timeout python run_freebase_magnn_nc_augmentation.py --dataset IMDB --group mode0 --variants "${VARIANTS// /,}" --seed "$SD" \
        --data-root "$BUNDLES" --output-root "$OUT" $EXTRA
      continue ;;
  esac
  for V in $VS; do
    [ -f "$OUT/$RA/IMDB/nc/$V/seed_$SD/metrics.json" ] && { echo "[skip] $RA $V seed_$SD"; continue; }
    _timeout python run_freebase_magnn_nc.py --dataset IMDB --arm "$RA" --variant "$V" --seed "$SD" --data-root "$BUNDLES" \
      --output-root "$OUT" --resume $EXTRA
  done
done
if [ "$RA" = augmentation ]; then aggregate "$OUT"; else aggregate "$OUT/$RA"; fi
