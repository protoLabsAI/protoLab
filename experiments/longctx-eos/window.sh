#!/usr/bin/env bash
# Guarded off-hours window for protoLab#36: PR #44993 overlay + penalty arms on the :8060 lane.
# Replica B leaves the pool for the run. Safety, learned on 2026-10-01 (replica A overloaded):
#   - guard: abort if A's waiting queue is > GUARD_WAIT for 3 consecutive minutes
#   - deadline: abort at DEADLINE (UTC) — traffic climbs from ~17Z
#   - trap: replica B is restored on ANY exit
set -uo pipefail
cd "$(dirname "$0")"
PY=~/dev/vllm-025/bin/python
OVERLAY=~/dev/vllm-025-pr44993-overlay
N=${N:-80}
GUARD_WAIT=${GUARD_WAIT:-8}
DEADLINE=${DEADLINE:-16:45}
ABORT=.window-abort
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
waiting() { curl -s -m 5 localhost:8041/metrics | awk '/^vllm:num_requests_waiting\{/ {print int($2)}'; }

kill_trials() { for p in $(ps -eo pid,args | grep '[t]rial.py' | awk '{print $1}'); do kill "$p"; done; }
stop_lane() {
  for p in $(ps -eo pid,args | grep '[v]llm serve .*--port 8060' | awk '{print $1}'); do kill "$p"; done
  until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 5; done
}
restore_b() {
  kill_trials; stop_lane
  log "restoring replica B"
  sudo systemctl reset-failed vllm-smart-qwen38-b.service
  sudo systemctl start vllm-smart-qwen38-b.service
  until curl -sf localhost:8042/health >/dev/null; do sleep 10; done
  log "replica B healthy on :8042 — confirm the gateway routes to it (homelab-iac#291)"
}
trap 'kill $GUARD_PID 2>/dev/null; restore_b; log "WINDOW END"' EXIT

rm -f $ABORT
(  # guard
  strikes=0
  while :; do
    sleep 60
    w=$(waiting); w=${w:-0}
    if [ "$w" -gt "$GUARD_WAIT" ]; then strikes=$((strikes+1)); else strikes=0; fi
    if [ $strikes -ge 3 ]; then echo "A waiting=$w for 3 min" > $ABORT; kill_trials; exit; fi
    if [[ "$(date -u +%H:%M)" > "$DEADLINE" ]]; then echo "deadline $DEADLINE" > $ABORT; kill_trials; exit; fi
  done
) & GUARD_PID=$!

log "WINDOW START — A waiting=$(waiting)"
sudo systemctl stop vllm-smart-qwen38-b.service
until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 5; done
OVERLAY=$OVERLAY SPEC_K=3 setsid nohup ./lane.sh >/dev/null 2>&1 &
until curl -sf localhost:8060/health >/dev/null; do sleep 10; [ -f $ABORT ] && exit 1; done
ENG=$(ps -eo pid,args | grep '[v]llm serve .*--port 8060' | awk '{print $1}' | head -1)
log "lane up (K=3 + PR #44993 overlay); PYTHONPATH in engine env: $(tr '\0' '\n' < /proc/$ENG/environ | grep -c "^PYTHONPATH=$OVERLAY")"

arm() {  # label extra-args...
  local label=$1; shift
  [ -f $ABORT ] && { log "skip $label: ABORT $(cat $ABORT)"; return; }
  log "arm $label start"
  $PY trial.py --base-url http://localhost:8060/v1 --label "$label" --n $N --conc 10 "$@" 2>&1 | grep -v -i warn
  log "arm $label done: $(grep -vc '"error"' results/$label.jsonl 2>/dev/null) rows"
}
arm p-k3                                  # patched baseline: JSON should be 0 invalid
arm p-k3-fp03  --frequency-penalty 0.3
arm p-k3-rp105 --repetition-penalty 1.05
[ -f $ABORT ] && log "ABORTED: $(cat $ABORT)"
