#!/bin/bash
# Quick progress view for this project's Slurm jobs (run on the submit node, from anywhere).
#   bash slurm/status.sh             # all my jobs: queue, last 24 h of accounting, latest log lines
#   bash slurm/status.sh 21955721    # one job, including every array task
#   N=30 bash slurm/status.sh        # show more log files (default 12 most recent)
set -uo pipefail
cd "$(dirname "$0")/.."
JOB=${1:-}

echo "== Queue (PENDING / RUNNING)"
squeue -u "$USER" ${JOB:+-j "$JOB"} -o "%.18i %.20j %.9T %.10M %.10l %R"

echo
echo "== Accounting (finished and running, last 24 h)"
sacct -u "$USER" ${JOB:+-j "$JOB"} -X -S now-1days \
  -o JobID%16,JobName%20,State%11,Elapsed,NodeList%12 | grep -v "^ *[0-9_]*\.\(batch\|extern\)"

echo
echo "== Latest progress line per log (logs/*.out, newest first)"
for log in $(ls -t logs/*${JOB}*.out 2>/dev/null | head -n "${N:-12}"); do
  # Training prints "Step k/n", evaluation prints "Saved ...: MRR=", failures print a Traceback/Error.
  last=$(grep -E "Step [0-9]+/[0-9]+|Saved |Error|Traceback|CANCELLED|DUE TO TIME LIMIT" "$log" | tail -1)
  printf "%-42s %s\n" "$(basename "$log")" "${last:-(started, no progress line yet)}"
done
