# Shared helpers for the MAGNN run scripts (sourced, not executed): environment, arm mapping,
# data-level hash check (gauntlet) and aggregation. Runs are skip-if-done and pass --resume.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAGNN="$RELEASE/models/MAGNN"
DATA="${INV_GNN_DATA:-$RELEASE/data}"
export INV_GNN_DATA="$DATA"
[ -n "${ENV_ACTIVATE:-}" ] && source "$ENV_ACTIVATE"
# determinism environment
export DGLBACKEND=pytorch OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS="${SEEDS:-1566911444 20241017 20251017}"
ARM="${ARM:?set ARM=original|universal|augmentation|invariant}"
FLAVOUR="${FLAVOUR:-union}"            # invariant arm only: union (runner arm invariant_union) | intersection (runner arm invariant)
RESULTS="${RESULTS:-$MAGNN/results}"
RUN_TIMEOUT="${RUN_TIMEOUT:-}"         # optional per-run wall limit for `timeout`, e.g. 4h (runs resume per epoch); empty = none
EXTRA="${EXTRA:-}"
cd "$MAGNN"

_timeout () { if [ -n "$RUN_TIMEOUT" ]; then timeout -k 30s "$RUN_TIMEOUT" "$@"; else "$@"; fi; }

# ARM -> the runner's internal arm name
runner_arm () {
  case "$ARM" in
    original) echo originals ;;
    universal) echo universal ;;
    augmentation) echo augmentation ;;
    invariant) case "$FLAVOUR" in union) echo invariant_union ;; intersection) echo invariant ;; *) echo "FLAVOUR must be union|intersection" >&2; exit 2 ;; esac ;;
    *) echo "ARM must be original|universal|augmentation|invariant" >&2; exit 2 ;;
  esac
}

# data-level gauntlet (program hashes equal across variants, supervision identical);
# GAUNTLET=0 skips it.  $1 = bundle dir of the task (contains invariant/ invariant_union/ stock/), $2 = variants, $3 = universal name
gauntlet () {
  [ "${GAUNTLET:-1}" = "0" ] && return 0
  D="$1" V="$2" U="$3" python3 - <<'PY' || { echo "GAUNTLET FAILED"; exit 9; }
import json, os
D = os.environ["D"]; V = os.environ["V"].split(); U = os.environ["U"]
inv = {v: json.load(open(f"{D}/invariant/{v}/summary.json")) for v in V}
invu = {v: json.load(open(f"{D}/invariant_union/{v}/summary.json")) for v in V}
st = {v: json.load(open(f"{D}/stock/{v}/summary.json")) for v in V + [U]}
ok = len({s["program_hash"] for s in inv.values()}) == 1 and len({s["program_hash"] for s in invu.values()}) == 1 \
     and len({s["supervision_hash"] for s in list(inv.values()) + list(invu.values()) + list(st.values())}) == 1
for v, s in inv.items(): print(f"  invariant {v:16s} program {s['program_hash'][:12]} physical {s['physical_hash'][:12]}")
for v, s in invu.items(): print(f"  inv_union {v:16s} program {s['program_hash'][:12]}")
for v, s in st.items(): print(f"  stock     {v:16s} program {s['program_hash'][:12]}")
print("GAUNTLET (data-level):", "PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
PY
}

aggregate () {  # $1 = results root that contains <DATASET>/<task>/<variant>/seed_*
  [ -d "$1" ] && python "$RELEASE/common/aggregate_rgcn_structural.py" --results-root "$1" --out-dir "$1/aggregate" 2>&1 | tail -1
}
