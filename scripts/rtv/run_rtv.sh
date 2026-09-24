#!/bin/bash
# Generate and score teacher answers, then filter by TAU.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ"

DATA=${DATA:-data/processed/train_pi_aspects.jsonl}
RAW=${RAW:-data/raw/train.jsonl}
TAU=${TAU:-0.6}
GEN_URL=${GEN_URL:?set GEN_URL to the teacher endpoint, e.g. http://127.0.0.1:8400/v1}
JUDGE_URL=${JUDGE_URL:?set JUDGE_URL to the judge endpoint, e.g. http://127.0.0.1:8500/v1}
GEN_MODEL=${GEN_MODEL:-policy}
JUDGE_MODEL=${JUDGE_MODEL:-judge}
MERGE_SYSTEM=${MERGE_SYSTEM:-}   # set to "--merge-system" for Gemma-2 data

[ -f "$DATA" ] || { echo "missing DATA: $DATA" >&2; exit 1; }

STEM=$(basename "${DATA%.jsonl}")
mkdir -p outputs/rtv
SCORES=outputs/rtv/${STEM}.scores.jsonl
OUT=data/processed/${STEM}_rtv$(printf '%03d' "$(python3 -c "print(int(float('$TAU')*100))")").jsonl

echo "=== [1/2] rtv_score -> $SCORES"
python src/rtv_score.py \
    --data "$DATA" \
    --raw "$RAW" \
    --out "$SCORES" \
    --gen-base-url "$GEN_URL"  --gen-model  "$GEN_MODEL" \
    --judge-base-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" \
    $MERGE_SYSTEM

echo "=== [2/2] rtv_filter (tau=$TAU) -> $OUT"
python src/rtv_filter.py \
    --data "$DATA" \
    --scores "$SCORES" \
    --raw "$RAW" \
    --tau "$TAU" \
    --out "$OUT"

echo "--- done"
echo "    scores   : $SCORES"
echo "    filtered : $OUT ($(wc -l < "$OUT") rows kept, tau=$TAU)"
echo "    next     : DATA=$OUT bash scripts/rtv/train_rtv_filtered.sh"
