#!/bin/bash
# Build the default rubric-aspects training dataset.
set -euo pipefail
PROJ=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RAW_DATA=${DATA:-data/raw}
STUDENT=${STUDENT:-Qwen/Qwen2.5-7B-Instruct}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJ"
mkdir -p data/processed

python src/build_data.py \
    --tokenizer "$STUDENT" \
    --max-prompt-tokens "${MAX_TOK:-6144}" \
    --shuffle \
    --input "$RAW_DATA/train.jsonl" \
    --output data/processed/train_pi_aspects.jsonl \
    --pi-mode aspects
