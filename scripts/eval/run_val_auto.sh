#!/bin/bash
# Evaluate checkpoints under RUN. Required: ARM, FAMILY, RUN; optional: SPLIT, LIMIT, NOJUDGE.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
ARM=${ARM:?set ARM}
FAMILY=${FAMILY:?set FAMILY=gemma|qwen25|qwen3}
RUN=${RUN:?set RUN=<run dir with checkpoint-*>}
PORT=${PORT:-8480}
SPLIT=${SPLIT:-val}
LIMIT=${LIMIT:-}
cd "$PROJ"

case "$SPLIT" in
  val)  EXP=2503 ;;
  test) EXP=2830 ;;
  *) echo "[FATAL] SPLIT must be val or test"; exit 1 ;;
esac
[ -n "$LIMIT" ] && EXP=$LIMIT

STEPS=$(ls -d $RUN/checkpoint-* 2>/dev/null | sed 's|.*/checkpoint-||' | sort -n | paste -sd' ' -)
[ -n "$STEPS" ] || { echo "[FATAL] no checkpoint-* under $RUN"; exit 1; }

LEDGER=/tmp/val_${ARM}_${SPLIT}.ledger
echo "===== arm=$ARM family=$FAMILY split=$SPLIT expect=$EXP ckpts=[$STEPS] =====" | tee -a $LEDGER

for STEP in $STEPS; do
  TAG=${ARM}_ckpt${STEP}
  OUT=outputs/${SPLIT}_${TAG}.json
  HAVE=$(python3 "$HERE/count_out.py" "$OUT" | awk '{print $1}')
  if [ "$HAVE" = "$EXP" ]; then
    echo "=== [$STEP] already complete ($HAVE rows), skipping ===" | tee -a $LEDGER
    continue
  fi

  echo "=== [$STEP] serving $(date +%H:%M:%S) ===" | tee -a $LEDGER
  FAMILY=$FAMILY bash "$HERE/serve_policy.sh" $RUN/checkpoint-${STEP} $PORT \
    || { echo "[FAIL] serve failed at step $STEP" | tee -a $LEDGER; exit 1; }

  OK=0
  for i in $(seq 1 60); do
    if curl -s -m 5 http://127.0.0.1:${PORT}/v1/models 2>/dev/null | grep -q policy; then
      OK=1; echo "  ready after ${i} polls" | tee -a $LEDGER; break
    fi
    sleep 20
  done
  if [ $OK -ne 1 ]; then
    echo "[FAIL] server never became ready at step $STEP" | tee -a $LEDGER
    tail -30 /tmp/policy_${PORT}.log
    tmux kill-session -t policy_${PORT} 2>/dev/null
    exit 1
  fi

  echo "=== [$STEP] generating $EXP rows ===" | tee -a $LEDGER
  PORT=$PORT TAG=$TAG FAMILY=$FAMILY SPLIT=$SPLIT LIMIT=$LIMIT bash "$HERE/gen_val_auto.sh"
  RC=$?
  N=$(python3 "$HERE/count_out.py" "$OUT" | awk '{print $1}')
  echo "=== [$STEP] rc=$RC n=$N ===" | tee -a $LEDGER

  tmux kill-session -t policy_${PORT} 2>/dev/null
  sleep 25   # let the four cards actually free before the next load

  if [ "$N" != "$EXP" ]; then
    echo "[FAIL] step $STEP incomplete: $N != $EXP, stopping" | tee -a $LEDGER
    exit 1
  fi
done

echo "===== generation done for arm=$ARM =====" | tee -a $LEDGER
ls -l outputs/${SPLIT}_${ARM}_ckpt*.json | awk '{print "  "$9" "$5" bytes"}' | tee -a $LEDGER

if [ -n "${NOJUDGE:-}" ] || [ -n "$LIMIT" ]; then
  echo "[info] skipping judge (NOJUDGE or LIMIT set)" | tee -a $LEDGER
  exit 0
fi

echo "===== judging arm=$ARM $(date +%H:%M:%S) =====" | tee -a $LEDGER
ARM=$ARM SPLIT=$SPLIT bash "$HERE/judge_val_auto.sh" 2>&1 | tee -a $LEDGER
