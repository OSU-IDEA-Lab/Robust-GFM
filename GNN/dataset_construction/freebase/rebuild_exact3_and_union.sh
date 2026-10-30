#!/bin/bash
# Run with `bash rebuild_exact3_and_union.sh`, or submit with sbatch after adapting the lines below.
#SBATCH -J fb_rebuild
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o fb_rebuild_%j.out

# Rebuild Freebase exact_3 from the clean range_2_3, then build the true union.
#
# Ordering is chosen so nothing is destroyed until its replacement exists and has
# passed an exact per-relation row-count check:
#   1. build exact_3/link.dat.new  (8.7 GB) from range_2_3   -> verify
#   2. only then replace the corrupt exact_3/link.dat (33.9 GB)
#   3. build union_exact_2_3/link.dat from the new exact_3 + exact_2 -> verify
#
# range_2_3 is never modified; it stays the semantic catalog.

set -euo pipefail
export LC_ALL=C

# ROOT = this folder (holds registries/; freebase_rebuild/ is written here). Under sbatch the script runs from a
# spool copy, so fall back to the submit dir; FB_REBUILD_ROOT overrides both.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ROOT/freebase_rebuild_exact3_and_union.py" ] || ROOT="${SLURM_SUBMIT_DIR:-$ROOT}"
ROOT="${FB_REBUILD_ROOT:-$ROOT}"
export INV_GNN_DATA="${INV_GNN_DATA:-$ROOT/../../data}"
D="$INV_GNN_DATA/dataset_variant_3hops_filtered"
M=$ROOT/freebase_rebuild
UNION=$D/union_exact_2_3
CORRUPT_BACKUP=$D/exact_3/link.dat.corrupt
cd "$ROOT"

free_gb() { df -BG "$D" | awk 'NR==2{gsub("G","",$4); print $4}'; }
counts_to_json() { sort -n "$1" | awk 'BEGIN{printf "{"} {printf "%s\"%s\": %s", (NR>1?", ":""), $1, $2} END{print "}"}' > "$2"; }
# NOTE: do NOT use `grep $'\x00'` for this. A grep pattern is a NUL-terminated C
# string, so the NUL makes it an EMPTY pattern that matches every line -- it
# reports corruption in every file, including clean ones.
nul_count() { LC_ALL=C tr -dc '\000' < "$1" | wc -c; }
# Recount rows per relation id straight from a built link.dat.
recount() { gawk -F'\t' -v C="$2" '{c[$3]++} END{for(r in c) printf "%s\t%s\n", r, c[r] > C}' "$1"; }

echo "=== preflight ==="
echo "free: $(free_gb)G"
[ "$(free_gb)" -ge 12 ] || { echo "FATAL: need >=12G free to stage exact_3" >&2; exit 3; }
python3 freebase_rebuild_exact3_and_union.py plan
test -s "$M/map_r23_to_exact3.tsv"

# --- 1. exact_3 from range_2_3 -----------------------------------------------
# Idempotent: if a previous run already produced and swapped in a good exact_3,
# re-verify it in place rather than repeating the 38.6 GB scan.
echo
if [ -s "$D/exact_3/link.dat" ] && [ ! -s "$D/exact_3/link.dat.new" ] \
   && [ "$(stat -c%s "$D/exact_3/link.dat")" -lt 20000000000 ]; then
  echo "=== [1/3] exact_3/link.dat already rebuilt; re-verifying in place ==="
  recount "$D/exact_3/link.dat" "$M/counts_exact_3.tsv"
else
  echo "=== [1/3] extracting exact_3 from range_2_3 (38.6 GB scan) ==="
  time gawk -F'\t' -v OFS='\t' -v COUNTS="$M/counts_exact_3.tsv" '
    NR==FNR { m[$1]=$2; next }
    ($3 in m) { $3=m[$3]; print; c[$3]++ }
    END { for (r in c) printf "%s\t%s\n", r, c[r] > COUNTS }
  ' "$M/map_r23_to_exact3.tsv" "$D/range_2_3/link.dat" > "$D/exact_3/link.dat.new"
  ls -l "$D/exact_3/link.dat.new"
fi

counts_to_json "$M/counts_exact_3.tsv" "$M/actual_counts_exact_3.json"
python3 freebase_rebuild_exact3_and_union.py verify-counts \
  --expected "$M/expected_counts_exact_3.json" --actual "$M/actual_counts_exact_3.json"

# --- 2. swap in (only after verification) -------------------------------------
echo
echo "=== [2/3] installing the verified exact_3/link.dat ==="
if [ -s "$D/exact_3/link.dat.new" ]; then
  ls -l "$D/exact_3/link.dat"
  # rename rather than unlink, so the corrupt original stays recoverable until
  # everything downstream has passed; released at the very end.
  mv "$D/exact_3/link.dat" "$CORRUPT_BACKUP"
  mv "$D/exact_3/link.dat.new" "$D/exact_3/link.dat"
fi
N=$(nul_count "$D/exact_3/link.dat")
if [ "$N" -ne 0 ]; then
  echo "FATAL: rebuilt exact_3/link.dat contains $N NUL bytes" >&2
  exit 5
fi
echo "[ok] rebuilt exact_3/link.dat is NUL-free and row-count verified"
echo "free: $(free_gb)G"

# --- 3. true union = base + exact_2's 6 + exact_3's 22 ------------------------
echo
echo "=== [3/3] building union_exact_2_3 (64 relations) ==="
mkdir -p "$UNION"
: > "$M/counts_union.tsv"
# emitted in relation-id order: base 0..35, then exact_2 36..41, then exact_3 42..63
gawk -F'\t' -v OFS='\t' -v COUNTS="$M/counts_union.tsv" \
  '$3 < 36 { print; c[$3]++ } END { for (r in c) printf "%s\t%s\n", r, c[r] >> COUNTS }' \
  "$D/exact_3/link.dat" > "$UNION/link.dat"
gawk -F'\t' -v OFS='\t' -v COUNTS="$M/counts_union.tsv" \
  '$3 >= 36 { print; c[$3]++ } END { for (r in c) printf "%s\t%s\n", r, c[r] >> COUNTS }' \
  "$D/exact_2/link.dat" >> "$UNION/link.dat"
gawk -F'\t' -v OFS='\t' -v COUNTS="$M/counts_union.tsv" \
  '$3 >= 36 { $3=$3+6; print; c[$3]++ } END { for (r in c) printf "%s\t%s\n", r, c[r] >> COUNTS }' \
  "$D/exact_3/link.dat" >> "$UNION/link.dat"

counts_to_json "$M/counts_union.tsv" "$M/actual_counts_union.json"
python3 freebase_rebuild_exact3_and_union.py verify-counts \
  --expected "$M/expected_counts_union.json" --actual "$M/actual_counts_union.json"

python3 freebase_rebuild_exact3_and_union.py union-info --union-dir "$UNION"
# node.dat / label.dat are identical across variants; share them rather than copy
for f in node.dat label.dat; do
  a=$(sha256sum "$D/unchanged/$f" | cut -d' ' -f1)
  b=$(sha256sum "$D/exact_3/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "FATAL: $f differs between variants" >&2; exit 6; }
  ln -f "$D/unchanged/$f" "$UNION/$f"
done
echo "[ok] node.dat/label.dat verified identical and hardlinked into the union dir"

# Everything verified: release the corrupt original.
rm -f "$CORRUPT_BACKUP"

echo
echo "=== done ==="
for v in unchanged exact_2 exact_3 range_2_3 union_exact_2_3; do
  printf "%-18s %14s bytes  %s relations\n" "$v" "$(stat -c%s $D/$v/link.dat)" \
    "$(awk '/^LINK\tSTART\tEND\tMEANING/{f=1;next} f&&NF==0{f=0} f&&/^[0-9]/{n++} END{print n+0}' $D/$v/info.dat)"
done
df -BG "$D" | tail -1
