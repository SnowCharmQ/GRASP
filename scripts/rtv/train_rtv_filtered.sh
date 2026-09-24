#!/bin/bash
# Train on RTV-filtered data with a fixed step budget.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJ"
mkdir -p logs

DATAFILE=${DATA:-$PROJ/data/processed/train_pi_aspects_rtv100.jsonl}
[ -f "$DATAFILE" ] || { echo "missing $DATAFILE (run scripts/rtv/run_rtv.sh first)" >&2; exit 1; }

SESS=rtv_train
tmux kill-session -t $SESS 2>/dev/null || true
TS=$(date +%m%d_%H%M)
tmux new-session -d -s $SESS "
  export DATA=$DATAFILE
  export OUT=$PROJ/outputs/rtv_train_$TS
  export LOG=logs/rtv_train_$TS.log
  export MASTER_PORT_OVR=29611
  export MAXSTEPS=${MAXSTEPS:-1000} SAVELIMIT=6
  bash scripts/train/train_qwen.sh
"
echo "--- launched $SESS"
echo "    data : $(basename $DATAFILE) ($(wc -l < $DATAFILE) rows)"
echo "    log  : logs/rtv_train_$TS.log  out: outputs/rtv_train_$TS"
