#!/bin/bash

set -eux

# Step 1: choose environment and constants
env="qinghai"



pip install deepspeed==0.17.4
pip install qwen_vl_utils==0.0.14
pip install -e /workspace/images-ks3-starfs-hd/workspace/lijiaze/projects/ms-swift-new/ms-swift-main

pip list

output_dir=""
cd "${code_dir}" || exit 1 


mkdir -p $output_dir

DECORD_EOF_RETRY_MAX=20480

# Step 4: create log file
#log_path="${output_dir}/${tag}.txt"
while true; do
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_path="${output_dir}/log_${timestamp}.txt"
    if [ -f "${log_path}" ]; then
        echo "Log file already exists: ${log_path}"
        sleep 1
    else
        echo "Creating log file: ${log_path}"
        break
    fi
done

echo '' > "${log_path}" 2>&1            # clear log file

# Step 5: display code version and other information
{
    echo "Code directory: ${code_dir}"
    echo "Code branch: $(git symbolic-ref --short HEAD)"
    echo "Code version: $(git rev-parse HEAD)"
} >> "${log_path}" 2>&1



export MODEL_SEQ_LEN=9103


train_cmd="swift rlhf \
--rlhf_type gkd \
--teacher_model $teacher_model_ckpt_path \
--output_dir $output_dir \
--model $policy_model_ckpt_path \
--train_type full \
--dataset $train_dataset.json \
--use_on_policy_distillation true \
--lmbda 1 \
--beta 1 \
--learning_rate 1e-6 \
--gradient_accumulation_steps 4 \
--freeze_llm false \
--freeze_vit true \
--freeze_aligner true \
--target_modules all-linear \
--max_completion_length 8192 \
--max_length 16000 \
--model_type qwen3_vl \
--teacher_model_type qwen3_vl \
--torch_dtype bfloat16 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--save_steps 5 \
--save_only_model false \
--save_total_limit 100 \
--logging_steps 1 \
--warmup_ratio 0.03 \
--dataloader_num_workers 64 \
--dataset_num_proc 4 \
--deepspeed zero3 \
--teacher_deepspeed zero3 \
--attn_impl flash_attn \
--use_vllm false "

# Step 7: run training
echo "Training command: ${train_cmd}"
tic=$(date +%s.%N)


WANDB_MODE=disabled DECORD_EOF_RETRY_MAX=20480 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=${WORLD_SIZE} NODE_RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} FPS_MAX_FRAMES=768 FPS_MIN_FRAMES=4 VIDEO_MAX_TOKEN_NUM=768 VIDEO_MIN_TOKEN_NUM=4 IMAGE_MAX_TOKEN_NUM=768 IMAGE_MIN_TOKEN_NUM=4 FPS=2 OMP_NUM_THREADS=16 ${train_cmd} >> "${log_path}" 2>&1

# Step 7: run training
echo "Training command: ${train_cmd}"
tic=$(date +%s.%N)

