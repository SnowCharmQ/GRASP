#!/bin/bash
# Serve the base teacher for RTV; use a base model to avoid circular filtering.
set -euo pipefail
CKPT=${1:?usage: serve_rtv.sh <teacher_ckpt_path> [port]}
PORT=${2:-8400}
GPUS=${GPUS:-0,1,2,3}
TP=${TP:-4}

[ -d "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }

unset MASTER_PORT MASTER_ADDR WORLD_SIZE RANK LOCAL_RANK

SESS=rtvgen_$PORT
tmux kill-session -t $SESS 2>/dev/null || true
tmux new-session -d -s $SESS "
  export CUDA_VISIBLE_DEVICES=$GPUS
  unset MASTER_PORT MASTER_ADDR WORLD_SIZE RANK LOCAL_RANK
  python -m vllm.entrypoints.openai.api_server \
    --model $CKPT \
    --served-model-name policy \
    --port $PORT \
    --tensor-parallel-size $TP \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --disable-log-requests \
    2>&1 | tee /tmp/rtvgen_$PORT.log
"
echo "launched $SESS -> port $PORT, tp $TP on GPU $GPUS"
echo "  ckpt: $CKPT"
echo "  log:  /tmp/rtvgen_$PORT.log"
