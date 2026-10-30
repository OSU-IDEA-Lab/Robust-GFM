# Internal: IMDb year graph, one task ($TASK = nc | movie_year | movie_director), one arm.
# original / universal / invariant: run_rgcn_structural.py; augmentation:
# run_rgcn_structural_{nc,lp}_augmentation.py.
# Bundle: data/preprocessed/RGCN_structural_v10_year_canonfeat (build with scripts/RGCN/preprocess_imdb.sh).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_arm
BUNDLE="${BUNDLE:-data/preprocessed/RGCN_structural_v10_year_canonfeat}"
case "$TASK" in
  movie_director) VARIANTS="v1 v3" ;;   # the director is the target; only the actor relocates
  *)              VARIANTS="v1 v2 v3 v4" ;;
esac
if [[ "$TASK" == nc ]]; then
  TRAIN_ARGS="--epochs 300 --patience 50"
  RUN_TIMEOUT="${RUN_TIMEOUT:-20m}"
else
  TRAIN_ARGS="--decoder dot --loss pairwise --select-on val_mrr --eval-protocol sampled --epochs 1000 --patience 200"
  RUN_TIMEOUT="${RUN_TIMEOUT:-45m}"
fi
OUT="$RESULTS/imdb_${TASK}/$ARM"
cd "$RGCN"
run_one () {  # $1 = --graph, $2 = variant
  for SD in $SEEDS; do
    if [[ -f "$OUT/IMDB/$TASK/$2/seed_$SD/metrics.json" ]]; then echo "[skip] $2 seed_$SD"; continue; fi
    timeout -k 30s "$RUN_TIMEOUT" python run_rgcn_structural.py \
      --data-root "$BUNDLE" --output-root "$OUT" \
      --dataset IMDB --task "$TASK" --variant "$2" --seed "$SD" \
      --device "$DEVICE" --graph "$1" --norm mean --emb-init 0.1 $TRAIN_ARGS
  done
}
case "$ARM" in
  invariant) for V in $VARIANTS; do run_one semantic "$V"; done ;;
  original)  for V in $VARIANTS; do run_one physical "$V"; done ;;
  universal) run_one physical universal ;;
  augmentation)
    for SD in $SEEDS; do
      if [[ "$TASK" == nc ]]; then
        [[ -f "$OUT/IMDB/nc/joint_seed_$SD.json" ]] && { echo "[skip] seed_$SD"; continue; }
        timeout -k 30s "${AUG_TIMEOUT:-30m}" python run_rgcn_structural_nc_augmentation.py \
          --data-root "$BUNDLE" --output-root "$OUT" --seed "$SD" --device "$DEVICE" \
          --norm mean --emb-init 0.1 --epochs 300 --patience 50
      else
        [[ -f "$OUT/IMDB/$TASK/joint_seed_$SD.json" ]] && { echo "[skip] seed_$SD"; continue; }
        timeout -k 30s "${AUG_TIMEOUT:-60m}" python run_rgcn_structural_lp_augmentation.py \
          --data-root "$BUNDLE" --output-root "$OUT" --task "$TASK" --variants "$(comma $VARIANTS)" \
          --seed "$SD" --device "$DEVICE" \
          --norm mean --emb-init 0.1 --decoder dot --loss pairwise --select-on val_mrr --epochs 1000 --patience 200
      fi
    done ;;
esac
echo "[done] IMDb $TASK $ARM -> $OUT"
