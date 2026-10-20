#!/bin/zsh
setopt pipefail  # fail on the Python exit status, not grep's
# Network-null analysis (paper/new_idea.md); logs to <run>/revision/nulls/logs/.
# Arguments (e.g. --run <name>, --overwrite) are forwarded to every script.
# Interpreter: $PYTHON (default: python on PATH).
# 01 resumes from finished chunks, so re-running after an interruption is safe.
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"
RUN_NAME="${HOC_RUN:-main}"
for ((i = 1; i <= $#; i++)); do
  [[ "${@[i]}" == "--run" ]] && RUN_NAME="${@[i+1]}"
  [[ "${@[i]}" == --run=* ]] && RUN_NAME="${@[i]#--run=}"
done
LOG="${HOC_RUNS_ROOT:-$REPO/runs}/$RUN_NAME/revision/nulls/logs"; mkdir -p "$LOG"
for s in 00_validate 01_run_nulls 02_analyze 03_gap_mediators 04_rayleigh_bound; do
  echo "START $s $(date +%T)"
  "${PYTHON:-python}" -u -W ignore $s.py "$@" 2>&1 | { grep -v "rewirable" || true; } > "$LOG/$s.log" || { echo "FAIL $s"; exit 1; }
  echo "END $s $(date +%T)"
done
