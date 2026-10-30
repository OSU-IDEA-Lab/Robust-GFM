#!/bin/bash
# MAGNN x Freebase NC (unchanged / exact_2 / exact_3 + universal union_exact_2_3).
#   ARM=original|universal|augmentation|invariant  [FLAVOUR=union|intersection]  [SEEDS="1566911444 20241017 20251017"]
# Prerequisite: scripts/MAGNN/preprocess_freebase.sh. Resources: one GPU (V100 32 GB), 4 CPU cores, 48 GB RAM;
# ~20 min-1.5 h per run. The augmentation runner has no --resume (~45 min per seed); the other arms resume per epoch.
# Example: ARM=invariant FLAVOUR=union scripts/MAGNN/run_freebase_nc.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BUNDLES="$DATA/preprocessed/freebase_magnn_v2"
VARIANTS="unchanged exact_2 exact_3"
OUT="$RESULTS/freebase_nc/$ARM"
RA="$(runner_arm)"
gauntlet "$BUNDLES" "$VARIANTS" union_exact_2_3
for SD in $SEEDS; do
  case "$RA" in
    invariant|invariant_union|originals) VS="$VARIANTS" ;;
    universal) VS="union_exact_2_3" ;;
    augmentation)
      if ls "$OUT/FREEBASE/nc"/*/seed_$SD/metrics.json >/dev/null 2>&1; then echo "[skip] augmentation seed_$SD"; continue; fi
      _timeout python run_freebase_magnn_nc_augmentation.py --dataset FREEBASE --seed "$SD" --data-root "$BUNDLES" --output-root "$OUT" $EXTRA
      continue ;;
  esac
  for V in $VS; do
    [ -f "$OUT/$RA/FREEBASE/nc/$V/seed_$SD/metrics.json" ] && { echo "[skip] $RA $V seed_$SD"; continue; }
    _timeout python run_freebase_magnn_nc.py --dataset FREEBASE --arm "$RA" --variant "$V" --seed "$SD" --data-root "$BUNDLES" \
      --output-root "$OUT" --resume $EXTRA
  done
done
if [ "$RA" = augmentation ]; then aggregate "$OUT"; else aggregate "$OUT/$RA"; fi
