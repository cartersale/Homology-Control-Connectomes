#!/bin/zsh
# Network-null analysis (paper/new_idea.md); logs to results/revision/nulls/logs/
# 01 resumes from finished chunks, so re-running after an interruption is safe.
cd "$(dirname "$0")"
LOG=../../../results/revision/nulls/logs; mkdir -p $LOG
for s in 00_validate 01_run_nulls 02_analyze 03_gap_mediators; do
  echo "START $s $(date +%T)"
  /opt/anaconda3/bin/python -u -W ignore $s.py 2>&1 | grep -v "rewirable" > $LOG/$s.log || { echo "FAIL $s"; exit 1; }
  echo "END $s $(date +%T)"
done
