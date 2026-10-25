#!/bin/bash
# Agentic evals: AR vs Uno on BFCL v3 (greedy, accuracy + speed) and SWE-agent trajectory replay (T=1.0, speed).
# Usage: loopuno/agentic/run_all.sh ADAPTER [ADAPTER...]   (run loopuno/agentic/prep.py first)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
O=$P/results/agentic; mkdir -p $O
DP=${DP:-8}
run() { python loopuno/agentic/generate.py --data data/agentic/$2.jsonl --out $O/$1.jsonl --lora $3 --block $4 --temperature $5 --max-tokens 4096 --dp $DP; }
for T in bfcl swe_replay; do
  [ $T = bfcl ] && TEMP=0 || TEMP=1.0
  run ${T}_ar $T $1 1 $TEMP
  for A in "$@"; do run ${T}_uno_$(basename $A) $T $A 8 $TEMP; done
done
for f in $O/bfcl_*.jsonl; do case $f in *.grade*|*.summary*) ;; *) python loopuno/agentic/grade_bfcl.py $f ;; esac; done
