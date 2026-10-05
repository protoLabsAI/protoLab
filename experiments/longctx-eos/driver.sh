#!/usr/bin/env bash
# Runs the #36 arms back to back on the :8060 test lane, then restores replica B.
set -uo pipefail
cd "$(dirname "$0")"
PY=~/dev/vllm-025/bin/python
N=40
log() { echo "[$(date +%H:%M:%S)] $*"; }
ok_rows() { [ -f "results/$1.jsonl" ] && grep -vc '"error"' "results/$1.jsonl" || echo 0; }

stop_lane() {
  for p in $(ps -eo pid,args | grep '[v]llm serve .*--port 8060' | awk '{print $1}'); do kill "$p"; done
  until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 5; done
  log "lane down, GPU0 free"
}
start_lane() {
  SPEC_K=$1 setsid nohup ./lane.sh >/dev/null 2>&1 &
  until curl -sf localhost:8060/health >/dev/null; do sleep 10; done
  log "lane up, SPEC_K=$1: $(grep -o 'num_spec_tokens=[0-9]*\|Selected Cutlass[A-Za-z0-9]*' /mnt/scratch/logs/longctx-eos-k$1.log | tail -2 | tr '\n' ' ')"
}
arm() {  # label extra-args...
  local label=$1; shift
  log "arm $label start"
  $PY trial.py --base-url http://localhost:8060/v1 --label "$label" --n $N --conc 10 "$@" 2>&1 | grep -v -i warn
  log "arm $label done: $(ok_rows "$label") rows"
}

# k3 was started by hand with --n 120: stop it at N rows, then fill any gaps in seeds 0..N-1
until [ "$(ok_rows k3)" -ge $N ]; do sleep 30; done
for p in $(ps -eo pid,args | grep '[t]rial.py .*--label k3 ' | awk '{print $1}'); do kill "$p"; done
sleep 5; arm k3
arm k3-noschema --no-schema
stop_lane; start_lane 0; arm k0
stop_lane; start_lane 1; arm k1
stop_lane
log "restoring replica B"
sudo systemctl reset-failed vllm-smart-qwen38-b.service
sudo systemctl start vllm-smart-qwen38-b.service
until curl -sf localhost:8042/health >/dev/null; do sleep 10; done
log "replica B healthy on :8042 — ALL DONE"
