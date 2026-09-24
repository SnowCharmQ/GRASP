
\
MLP_WORKER_NUM=1
MLP_WORKER_GPU=8
MLP_ROLE_INDEX=0
MLP_WORKER_0_HOST=${MASTER_ADDR}
MLP_WORKER_0_PORT=20333


export PYTHONPATH=$(pwd)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MODEL_PATH=$ckpt_path
SPLIT="${SPLIT:-default}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/lvbench_ts_ablation_timelens}"
FPS="${FPS:-2}"
USE_VLLM="${USE_VLLM:-}" # set to any non-empty value to enable --use_vllm

TOTAL_GPUS="${TOTAL_GPUS:-8}"  # Total number of GPUs available (default: 8)
START_GPU_ID="${START_GPU_ID:-0}"  # Starting GPU ID (default: 0, so GPUs are 0-7)

# Automatically calculate GPU groups based on tensor parallel size
# Example: 8 GPUs, TP=2 → 4 groups: [0,1], [2,3], [4,5], [6,7]
# Example: 8 GPUs, TP=4 → 2 groups: [0,1,2,3], [4,5,6,7]
# Example: 8 GPUs, TP=8 → 1 group: [0,1,2,3,4,5,6,7]

if [[ $((TOTAL_GPUS % TENSOR_PARALLEL_SIZE)) -ne 0 ]]; then
    echo "Error: TOTAL_GPUS ($TOTAL_GPUS) must be divisible by TENSOR_PARALLEL_SIZE ($TENSOR_PARALLEL_SIZE)" >&2
    exit 1
fi
NUM_GROUPS=$((TOTAL_GPUS / TENSOR_PARALLEL_SIZE))
echo "Configuration:"
echo "  Total GPUs: $TOTAL_GPUS (IDs: $START_GPU_ID-$((START_GPU_ID + TOTAL_GPUS - 1)))"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Data Parallel Groups: $NUM_GROUPS"
echo "  GPUs per group: $TENSOR_PARALLEL_SIZE"

EXTRA_ARGS=()
if [[ -n "$USE_VLLM" ]]; then
  EXTRA_ARGS+=("--use_vllm")
  EXTRA_ARGS+=("--tensor_parallel_size" "$TENSOR_PARALLEL_SIZE")
fi


out_dir="$MODEL_PATH/eval_results_no_think/qvhighlights"

mkdir -p $out_dir

echo ""
echo "Launching $NUM_GROUPS data parallel processes..."
for ((shard=0; shard<NUM_GROUPS; shard++)); do
    # Build GPU group string directly
    START_ID=$((START_GPU_ID + shard * TENSOR_PARALLEL_SIZE))
    GPU_GROUP=""
    for ((j=0; j<TENSOR_PARALLEL_SIZE; j++)); do
        GPU_ID=$((START_ID + j))
        if [ -z "$GPU_GROUP" ]; then
            GPU_GROUP="$GPU_ID"
        else
            GPU_GROUP="$GPU_GROUP,$GPU_ID"
        fi
    done
    echo "Starting shard $shard on GPUs: $GPU_GROUP"
    CUDA_VISIBLE_DEVICES=$GPU_GROUP python eval_qwen3vl.py \
        --dataset qvhighlights \
        --model_path "$MODEL_PATH" \
        --split "$SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_jsonl "$out_dir"/qvhighlights_shard_${shard}_of_${NUM_GROUPS}.jsonl \
        --curr_idx $shard \
        --total_idx $NUM_GROUPS \
        --use_timelens \
        --fps "$FPS" \
        --no_duration_in_prompt \
        --nothink "True" \
        "${EXTRA_ARGS[@]}" &
done
wait

python eval/vllm_inference/eval_all.py --eval_root "$out_dir" --split "$SPLIT" --dataset qvhighlights 
