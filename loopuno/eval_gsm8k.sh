#!/bin/bash
# gsm8k (first 200 problems): TPF, tok/s and accuracy for each checkpoint at block 8, plus an AR (block 1) reference.
# Usage: loopuno/eval_gsm8k.sh CKPT_ROOT [steps...]   (default steps: 8 16 32 60)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DATA=$P/data/eval/gsm8k_from_openai.jsonl
if [ ! -s $DATA ]; then
  mkdir -p $(dirname $DATA)
  python -c "
import json
from evaluation.data import prepare_gsm8k
with open('$DATA', 'w') as f:
    for r in prepare_gsm8k(): f.write(json.dumps(r) + '\\n')
"
fi
ROOT=$1; shift; STEPS=${@:-8 16 32 60}
OUT=$P/results/gsm8k/$(basename $ROOT)
eval1() {  # name lora_dir block
  local D=$OUT/$1; mkdir -p $D
  echo "=== $1 start $(date -u +%FT%TZ)"
  python -m evaluation.run --benchmark gsm8k --model $UNO_MODEL_PATH --gated-lora-path $2 \
    --mask-token-id 166107 --stop-token-ids 166101 \
    --data $DATA --protocol-arm loopuno --limit ${LIMIT:-200} --skip-row-count-check \
    --max-model-len 32768 --max-num-batched-tokens 32768 --max-tokens 8192 --num-samples 1 \
    --data-parallel-size 8 --attention-backend fa2 --diffusion-block-size $3 \
    --output $D/generations.jsonl --summary-output $D/generation_summary.json --grades $D/grades.jsonl --scores $D/scores.json \
    2>&1 | grep -v Generating
  echo "=== $1 end $(date -u +%FT%TZ)"
}
LAST=$(ls -d $ROOT/checkpoint-* | sort -t- -k2 -n | tail -1)
[ -n "${SKIP_AR:-}" ] || eval1 ar-b1 $LAST 1
for s in $STEPS; do [ -d $ROOT/checkpoint-$s ] && eval1 step$s-b8 $ROOT/checkpoint-$s 8; done
python3 - $OUT <<'PY'
import json,sys,glob,os
for d in sorted(glob.glob(sys.argv[1]+"/*/")):
    try: g=json.load(open(d+"generation_summary.json")); s=json.load(open(d+"scores.json"))
    except Exception as e: print(os.path.basename(d[:-1]),"missing",e); continue
    print(f"{os.path.basename(d[:-1]):12s} acc={s['accuracy']:.3f} TPF={g['decoder_tokens_per_sequence_forward']:.2f} tok/s={g['output_tokens_per_second']:.0f} out_tok={g['total_output_tokens']}")
PY
