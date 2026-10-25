#!/bin/bash
# Train the LoopUno gated LoRA on Nanbeige4.2-3B (22 layers x 2 loops), block size 8, 8 GPUs.
# Defaults reproduce the one-hour run: 60 steps of 0.52M tokens at lr 1e-4.
# Continue a run: INIT=checkpoints/RUN/checkpoint-60 CURRICULUM=loopuno/curriculum_b8_cont180.yaml
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$P/.cache/wandb PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNO_SAVE_STEPS=${SAVE_STEPS:-8,16,32}
mkdir -p $WANDB_DIR
PER_DEVICE=${PER_DEVICE:-2}
RUN=${RUN:-loopuno-b8-lr1e-4}
torchrun --standalone --nproc-per-node=${NPROC:-8} -m training.train \
  --dataset-path $P/data/openthoughts-nanbeige-60k \
  --output-dir $P/checkpoints/$RUN \
  --curriculum ${CURRICULUM:-$P/loopuno/curriculum_b8_60.yaml} \
  --deepspeed $P/loopuno/deepspeed_zero0.json \
  --per-device-batch-size $PER_DEVICE \
  --gradient-accumulation-steps $((16 / PER_DEVICE)) \
  --learning-rate ${LR:-1e-4} --warmup-steps ${WARMUP:-3} ${INIT:+--init-adapter $INIT} \
  --lora-target all --lora-rank 128 --lora-alpha 2048 \
  --ce-alpha 0 --kl-beta 0 --tv-gamma 1 \
  --logging-steps 1 --run-name $RUN --wandb-mode $WANDB_MODE
