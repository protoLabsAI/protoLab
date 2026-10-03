#!/usr/bin/env bash
# Ornith-1.5-35B-A3B NVFP4 — the dual-GPU window.
#
# This STOPS PROD (vllm-smart-ornith15 = protolabs/{smart,reasoning,coder}) for the duration.
# Restore is in a trap so the lane comes back on success, failure, or kill -- the one thing
# that must never happen is finishing the night with prod still down.
set -uo pipefail
LOG=/mnt/scratch/logs/ornith15-35b-nvfp4.log
UNIT=vllm-smart-ornith15.service

restore() {
  echo "=== RESTORING $UNIT $(date -u +%H:%M:%S)"
  sudo systemctl start "$UNIT"
  for i in $(seq 1 90); do
    if curl -s -m 3 http://localhost:8041/v1/models >/dev/null 2>&1; then
      echo "=== prod back up $(date -u +%H:%M:%S): $(curl -s http://localhost:8041/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])')"
      return 0
    fi
    sleep 10
  done
  echo "!!! PROD DID NOT COME BACK -- investigate $UNIT"
}
trap restore EXIT INT TERM

echo "=== stopping $UNIT $(date -u +%H:%M:%S)"
sudo systemctl stop "$UNIT"
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 2p)
  [ "$used" -lt 30000 ] && break
  sleep 5
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== quantizing $(date -u +%H:%M:%S) (expect ~85 min at 128 samples)"
cd /home/ava/dev/lab/experiments/quantize
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ava/dev/quant-env/bin/python ornith15_35b_nvfp4_requant.py
rc=$?
echo "=== quant exit=$rc $(date -u +%H:%M:%S)"
