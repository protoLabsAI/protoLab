#!/usr/bin/env bash
# Guarded quiet-window training run on GPU0 (replica B out). NOT scheduled — review first.
# Same guards as experiments/longctx-eos/window.sh (2026-10-01 lesson: one replica alone can wedge):
#   abort if A's waiting queue > GUARD_WAIT for 3 consecutive minutes, abort at DEADLINE (UTC),
#   and restore replica B on ANY exit. Training is checkpoint-free for now: an abort loses the run.
#   RUNS="q08b:Qwen/Qwen3.5-0.8B q2b:Qwen/Qwen3.5-2B" CAP=2000 ./train-window.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=~/dev/quant-env/bin/python
RUNS=${RUNS:-"q08b:Qwen/Qwen3.5-0.8B q2b:Qwen/Qwen3.5-2B"}
CAP=${CAP:-2000}
LR=${LR:-1e-4}
GUARD_WAIT=${GUARD_WAIT:-8}
DEADLINE=${DEADLINE:-16:45}
ABORT=.train-abort
export HF_HOME=/mnt/models/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
waiting() { curl -s -m 5 localhost:8041/metrics | awk '/^vllm:num_requests_waiting\{/ {print int($2)}'; }
kill_train() { for p in $(ps -eo pid,args | grep '[t]rain_s1.py' | awk '{print $1}'); do kill "$p"; done; }
restore_b() {
  kill_train; sleep 10
  log "restoring replica B"
  sudo systemctl reset-failed vllm-smart-qwen38-b.service
  sudo systemctl start vllm-smart-qwen38-b.service
  until curl -sf localhost:8042/health >/dev/null; do sleep 10; done
  log "replica B healthy — confirm the gateway routes to it within ~2 min (homelab-iac#291)"
}
trap 'kill $GUARD_PID 2>/dev/null; restore_b; log "WINDOW END"' EXIT
rm -f $ABORT
( strikes=0
  while :; do
    sleep 60
    w=$(waiting); w=${w:-0}
    if [ "$w" -gt "$GUARD_WAIT" ]; then strikes=$((strikes+1)); else strikes=0; fi
    if [ $strikes -ge 3 ]; then echo "A waiting=$w for 3 min" > $ABORT; kill_train; exit; fi
    if [[ "$(date -u +%H:%M)" > "$DEADLINE" ]]; then echo "deadline $DEADLINE" > $ABORT; kill_train; exit; fi
  done ) & GUARD_PID=$!

log "WINDOW START — A waiting=$(waiting)"
sudo systemctl stop vllm-smart-qwen38-b.service
until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 5; done
for spec in $RUNS; do
  name=${spec%%:*}; base=${spec#*:}
  [ -f $ABORT ] && { log "skip $name: ABORT $(cat $ABORT)"; continue; }
  log "train $name ($base) cap=$CAP lr=$LR"
  $PY train_s1.py --base "$base" --out runs/$name-cap$CAP --device cuda --train-cap $CAP --lr $LR 2>&1 | grep -v -i -E 'warn|fast path'
  [ -f runs/$name-cap$CAP/adapter_model.safetensors ] && log "$name saved" || log "$name did not finish"
done
if [ -f $ABORT ]; then log "ABORTED: $(cat $ABORT)"; exit 1; fi
