#!/usr/bin/env bash
# Roll the smart lane back from Ornith-1.5-35B (ours) to Qwen3.8-27B-NVFP4+MTP.
#
# Why: measured board, same harness — Qwen3.8-27B-NVFP4-MTP claw 0.761 / LCB 0.632 vs
# Ornith-1.5-35B claw 0.719 / LCB 0.205. `coder` is one of the three aliases this lane
# serves, and Ornith-1.5's coding weakness is a property of the MODEL (upstream's own build
# scores LCB 0.192 on the same harness), not of our quant. The Ornith artifact stays
# published; it is just not the right prod lane.
#
# Reverse: systemctl disable --now vllm-smart-qwen38 && systemctl enable --now vllm-smart-ornith15
set -uo pipefail
echo "=== stopping ornith15 $(date -u +%H:%M:%S)"
sudo systemctl disable --now vllm-smart-ornith15.service 2>&1 | tail -2
echo "=== starting qwen38 $(date -u +%H:%M:%S)"
sudo systemctl enable --now vllm-smart-qwen38.service 2>&1 | tail -2
for i in $(seq 1 90); do
  if curl -s -m 3 http://localhost:8041/v1/models >/dev/null 2>&1; then
    R=$(curl -s -m 120 http://localhost:8041/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"smart","max_tokens":2048,"messages":[{"role":"user","content":"Reply with exactly: ROLLBACK OK"}]}' 2>/dev/null \
      | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print(((m.get('content') or '').strip() or (m.get('reasoning') or '').strip())[:60])" 2>/dev/null)
    if [ -n "$R" ]; then
      echo "=== REAL COMPLETION: $R  ($(date -u +%H:%M:%S))"
      curl -s http://localhost:8041/v1/models | python3 -c "import json,sys;print('=== aliases:',[m['id'] for m in json.load(sys.stdin)['data']])"
      exit 0
    fi
  fi
  sleep 10
done
echo "!!! qwen38 lane did not come up — check: journalctl -u vllm-smart-qwen38"
exit 1
