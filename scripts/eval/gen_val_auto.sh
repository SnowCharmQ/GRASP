#!/bin/bash
# Generate validation/test answers using an already-served policy.
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${PORT:-8480}
TAG=${TAG:?set TAG, e.g. q3_opsd_ckpt200}
FAMILY=${FAMILY:?set FAMILY=gemma|qwen25|qwen3}
SPLIT=${SPLIT:-val}
LIMIT=${LIMIT:-}

cd "$PROJ"
export PYTHONPATH=$PROJ/eval_src/src

case "$SPLIT" in
  val)  DATA=data/raw/validation.jsonl ;;
  test) DATA=data/raw/test.jsonl ;;
  *) echo "[FATAL] SPLIT must be val or test, got $SPLIT"; exit 1 ;;
esac

ARGS=""
case "$FAMILY" in
  gemma)  ARGS="--merge_system"; MAXTOK=${MAXTOK:-1024} ;;
  qwen25) MAXTOK=${MAXTOK:-2048} ;;
  qwen3)  MAXTOK=${MAXTOK:-2048}; [ -n "${NOTHINK:-}" ] && ARGS="--no_thinking" ;;
  *) echo "[FATAL] unknown FAMILY=$FAMILY"; exit 1 ;;
esac
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

OUT=outputs/${SPLIT}_${TAG}.json
echo "[gen] tag=$TAG family=$FAMILY split=$SPLIT max_tokens=$MAXTOK args='$ARGS'"

python eval_src/src/generate_responses.py \
  --mode official_rag \
  --dataset "$DATA" \
  --output "$OUT" \
  --model policy \
  --base_url "http://127.0.0.1:${PORT}/v1" \
  --num_contexts 10 \
  --temperature 0.1 \
  --top_p 0.95 \
  --max_tokens "$MAXTOK" \
  --concurrency 256 \
  $ARGS
RC=$?

read N EMPTY <<<"$(python3 "$HERE/count_out.py" "$OUT")"
echo "[gen] rc=$RC n=$N empty=$EMPTY -> $OUT"
exit $RC
