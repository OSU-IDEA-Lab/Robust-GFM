# Shared LP body (sourced by run_imdb_movie_year.sh / run_imdb_movie_director.sh / run_dblp_lp.sh).
# Needs: DATASET (IMDB|DBLP), TASK, VARIANTS, BUNDLES (root containing <TASK>/... for IMDB, or the dblp_magnn_v2 root), OUT, RA.
for SD in $SEEDS; do
  case "$RA" in
    invariant|invariant_union|originals) VS="$VARIANTS" ;;
    universal) VS="universal" ;;
    augmentation)
      if [ -f "$OUT/$DATASET/$TASK/v1/seed_$SD/metrics.json" ]; then echo "[skip] augmentation seed_$SD"; continue; fi
      _timeout python run_magnn_lp_augmentation.py --dataset "$DATASET" --task "$TASK" --variants "${VARIANTS// /,}" --seed "$SD" \
        --data-root "$BUNDLES" --output-root "$OUT" --resume $EXTRA
      continue ;;
  esac
  for V in $VS; do
    [ -f "$OUT/$RA/$DATASET/$TASK/$V/seed_$SD/metrics.json" ] && { echo "[skip] $RA $V seed_$SD"; continue; }
    _timeout python run_magnn_lp.py --dataset "$DATASET" --task "$TASK" --arm "$RA" --variant "$V" --seed "$SD" --data-root "$BUNDLES" \
      --output-root "$OUT" --resume $EXTRA
  done
done
if [ "$RA" = augmentation ]; then aggregate "$OUT"; else aggregate "$OUT/$RA"; fi
