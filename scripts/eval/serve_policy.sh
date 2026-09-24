#!/bin/bash
# Serve a policy checkpoint on GPUs 0-3; FAMILY sets the context limit.
set -u
CKPT=${1:?usage: FAMILY=gemma|qwen25|qwen3 serve_policy.sh <ckpt_path> [port]}
PORT=${2:-8480}
FAMILY=${FAMILY:?set FAMILY=gemma|qwen25|qwen3}
GPUS=${GPUS:-0,1,2,3}
TP=${TP:-4}

case "$FAMILY" in
  gemma)  MAXLEN=${MAXLEN:-8192}  ;;
  qwen25) MAXLEN=${MAXLEN:-16384} ;;
  qwen3)  MAXLEN=${MAXLEN:-16384} ;;
  *) echo "[FATAL] unknown FAMILY=$FAMILY (expected gemma|qwen25|qwen3)"; exit 1 ;;
esac

[ -f "$CKPT/config.json" ] || { echo "[FATAL] no checkpoint at $CKPT"; exit 1; }

SESS=policy_$PORT
tmux kill-session -t $SESS 2>/dev/null
tmux new-session -d -s $SESS "
  export CUDA_VISIBLE_DEVICES=$GPUS
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  unset MASTER_PORT MASTER_ADDR WORLD_SIZE RANK LOCAL_RANK
  python -m vllm.entrypoints.openai.api_server \\
    --model $CKPT --served-model-name policy --port $PORT \\
    --tensor-parallel-size $TP --gpu-memory-utilization 0.90 \\
    --max-model-len $MAXLEN --dtype bfloat16 --disable-log-requests \\
    2>&1 | tee /tmp/policy_$PORT.log
"
echo "launched $SESS: family=$FAMILY maxlen=$MAXLEN tp=$TP gpus=$GPUS port=$PORT"
echo "  ckpt: $CKPT"
echo "  log:  /tmp/policy_$PORT.log"
