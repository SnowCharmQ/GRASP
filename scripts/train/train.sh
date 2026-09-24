#!/bin/bash
# Shared GRASP training recipe; override paths and hyperparameters via environment.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE=${ROOT:-$PROJ}/train_engine
FAMILY=${1:?usage: train.sh qwen|gemma}
case "$FAMILY" in
  qwen) MODEL=${MODEL:-${MODELS:-$HOME/models}/Qwen2.5-7B-Instruct}
        DEFAULT_DATA=train_pi_aspects.jsonl; ATTN=flash_attn ;;
  gemma) MODEL=${MODEL:-${MODELS:-$HOME/models}/gemma-2-9b-it}
         DEFAULT_DATA=gemma_train_pi_aspects.jsonl; ATTN=${ATTN:-eager} ;;
  *) echo "Unknown family: $FAMILY" >&2; exit 1 ;;
esac
cd "$PROJ" || exit 1

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ENGINE:$PROJ/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LAMPQA_GRASP=1
export LAMPQA_GRASP_LOSS=1
export LAMPQA_GRASP_LOSS_BETA=${LAMPQA_GRASP_LOSS_BETA:-${LAMPQA_OPSD_BETA:-0}}          # 0 = forward KL
export LAMPQA_GRASP_LOSS_TEMPERATURE=${LAMPQA_GRASP_LOSS_TEMPERATURE:-${LAMPQA_OPSD_TEMPERATURE:-1.0}}
export LAMPQA_GRASP_LOSS_TOP_K=${LAMPQA_GRASP_LOSS_TOP_K:-${LAMPQA_OPSD_TOP_K:-0}}        # 0 = full vocabulary
export LAMPQA_GRASP_LOSS_TOKEN_CLIP=${LAMPQA_GRASP_LOSS_TOKEN_CLIP:-${LAMPQA_OPSD_TOKEN_CLIP:-0}}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
unset MASTER_PORT MASTER_ADDR WORLD_SIZE RANK LOCAL_RANK

DATA=${DATA:-$PROJ/data/processed/${DEFAULT_DATA}}
OUT=${OUT:-$PROJ/outputs/grasp_${FAMILY}_$(date +%m%d_%H%M)}
LOG=${LOG:-logs/train_${FAMILY}.log}
mkdir -p "$OUT" "$(dirname "$LOG")"

[ -f "$MODEL/config.json" ] || { echo "[FATAL] no model at $MODEL"; exit 1; }
[ -f "$DATA" ]              || { echo "[FATAL] no data at $DATA"; exit 1; }

echo "=== train $FAMILY | GPUs=$CUDA_VISIBLE_DEVICES model=$MODEL data=$DATA"

torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_port "${MASTER_PORT_OVR:-29537}" \
    src/launch_grasp.py \
    --rlhf_type gkd \
    --model "$MODEL" \
    --teacher_model "${TEACHER:-$MODEL}" \
    --dataset "$DATA" \
    --output_dir "$OUT" \
    --train_type full \
    --lmbda 1 --beta 0 \
    --learning_rate ${LR:-5e-6} \
    --lr_scheduler_type ${SCHED:-cosine} \
    --warmup_ratio ${WARMUP:-0.1} \
    --weight_decay ${WD:-0.1} \
    --per_device_train_batch_size ${PDBS:-1} \
    --gradient_accumulation_steps ${GAS:-4} \
    --max_completion_length ${MAXCOMP:-512} \
    --max_length ${MAXLEN:-6144} \
    --torch_dtype bfloat16 \
    --num_train_epochs ${EPOCHS:-1} \
    --max_steps ${MAXSTEPS:--1} \
    --save_steps ${SAVESTEPS:-200} \
    --save_only_model ${SAVEONLY:-true} \
    --save_total_limit ${SAVELIMIT:--1} \
    --logging_steps ${LOGSTEPS:-10} \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2 \
    --deepspeed zero3 \
    --teacher_deepspeed zero3 \
    --attn_impl "$ATTN" \
    --use_vllm false \
    --padding_free false \
    --seq_kd false \
    --fsdp '' \
    2>&1 | tee "$LOG"


if ! grep -q '\[GRASP\] compute_loss patched' "$LOG"; then
    echo "[warn] GRASP loss activation missing; inspect $LOG" >&2
fi
