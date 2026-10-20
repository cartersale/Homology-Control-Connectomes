#!/bin/zsh
# Runs all regime revision analyses in dependency order; logs to <run>/revision/regime/logs/.
# Arguments (e.g. --run <name>, --overwrite) are forwarded to every script.
# Interpreter: $PYTHON (default: python on PATH).
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"
RUN_NAME="${HOC_RUN:-main}"
for ((i = 1; i <= $#; i++)); do
  [[ "${@[i]}" == "--run" ]] && RUN_NAME="${@[i+1]}"
  [[ "${@[i]}" == --run=* ]] && RUN_NAME="${@[i]#--run=}"
done
LOG="${HOC_RUNS_ROOT:-$REPO/runs}/$RUN_NAME/revision/regime/logs"; mkdir -p "$LOG"
for s in 00_validate 01_csweep_perron 02_nctpy_equivalence 03_horizon_eps_largek 04_energy_floor 05_summarize_csweep 06_perron_diagnostic 08_perron_mode_sweep 07_report_tables 09_make_csweep_figure; do
  echo "START $s $(date +%T)"
  "${PYTHON:-python}" -u -W ignore::RuntimeWarning $s.py "$@" > "$LOG/$s.log" 2>&1 || { echo "FAIL $s"; exit 1; }
  echo "END $s $(date +%T)"
done
