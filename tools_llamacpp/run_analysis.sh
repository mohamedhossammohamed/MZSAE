#!/bin/bash
# 9-way parallel offline analysis, one process per capture. 10-min cap each.
for spec in "docA 8192" "docB 8192" "docC 8192" "docD 8192" "docA 16384" "docB 16384" "docC 16384" "docD 16384" "docA 32768"; do
  set -- $spec
  timeout 600 .venv/bin/python analyze_q4.py $1 $2 > logs/analyze_$1_$2.log 2>&1 &
done
wait
echo "ANALYSIS BATCH DONE"
grep -h "VERDICT\|clearing\|BLOCK PRUNING" logs/analyze_*.log 2>/dev/null | head
