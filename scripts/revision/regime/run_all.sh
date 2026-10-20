#!/bin/zsh
# Runs all regime revision analyses in order; logs to results/revision/regime/logs/
cd "$(dirname "$0")"
LOG=../../../results/revision/regime/logs; mkdir -p $LOG
for s in 00_validate 01_csweep_perron 02_nctpy_equivalence 03_horizon_eps_largek 04_energy_floor; do
  echo "START $s $(date +%T)"
  /opt/anaconda3/bin/python -u -W ignore::RuntimeWarning $s.py > $LOG/$s.log 2>&1 || { echo "FAIL $s"; exit 1; }
  echo "END $s $(date +%T)"
done
