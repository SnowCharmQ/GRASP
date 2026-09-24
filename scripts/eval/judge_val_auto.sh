#!/bin/bash
# Score one arm with a single judge load; defaults to GPUs 4-7.
set -uo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM=${ARM:?set ARM, e.g. q3_opsd}
SPLIT=${SPLIT:-val}
JUDGE="${JUDGE:-${MODELS:-$HOME/models}/Qwen2.5-32B-Instruct}"
JGPUS=${JGPUS:-4,5,6,7}
JTP=${JTP:-4}

cd "$PROJ"
unset MASTER_PORT MASTER_ADDR WORLD_SIZE RANK LOCAL_RANK
export CUDA_VISIBLE_DEVICES=$JGPUS
export PYTHONPATH=$PROJ/eval_src/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

case "$SPLIT" in
  val)  DATA=data/raw/validation.jsonl ;;
  test) DATA=data/raw/test.jsonl ;;
  *) echo "[FATAL] SPLIT must be val or test, got $SPLIT"; exit 1 ;;
esac

RESP=$(ls -1 outputs/${SPLIT}_${ARM}_ckpt*.json 2>/dev/null \
       | sed 's/.*_ckpt\([0-9]*\)\.json/\1 &/' | sort -n | awk '{print $2}' | paste -sd' ' -)
[ -n "$RESP" ] || { echo "[FATAL] no outputs/${SPLIT}_${ARM}_ckpt*.json found"; exit 1; }
echo "[judge] arm=$ARM split=$SPLIT gpus=$JGPUS tp=$JTP"
for f in $RESP; do echo "    $f"; done

OUTDIR=outputs/judge_${ARM}_${SPLIT}
python eval_src/src/official_eval.py \
  --dataset "$DATA" \
  --responses $RESP \
  --out_dir "$OUTDIR" \
  --evaluator_llm "$JUDGE" \
  --tensor_parallel_size $JTP \
  --max_length 32000
RC=$?

echo "[judge] rc=$RC -> $OUTDIR"
for f in $OUTDIR/*.scores.json; do echo "-- $f"; cat "$f"; echo; done
exit $RC
