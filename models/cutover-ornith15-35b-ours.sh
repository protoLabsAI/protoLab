#!/usr/bin/env bash
# Point the prod smart lane at OUR compressed-tensors NVFP4 build instead of upstream's
# ModelOpt one. Reversible: restore the .pre-ourquant-* unit backup and daemon-reload.
#
# Difference that matters if this ever misbehaves: upstream is ModelOpt MIXED_PRECISION
# with FP8 KV and it quantizes linear_attn.out_proj; ours is compressed-tensors
# nvfp4-pack-quantized with ALL of linear_attn left bf16 (the DeltaNet finding). Ours is
# 25.0 GB vs upstream's 23.4 GB for that reason.
set -euo pipefail
UNIT=/etc/systemd/system/vllm-smart-ornith15.service
NEW=/mnt/models/quantized/Ornith-1.5-35B-A3B-NVFP4

grep -q "^Environment=MODEL=" "$UNIT" || { echo "no MODEL= line in $UNIT"; exit 1; }
sudo sed -i "s|^Environment=MODEL=.*|Environment=MODEL=$NEW|" "$UNIT"
sudo systemctl daemon-reload
sudo systemctl restart vllm-smart-ornith15.service

for i in $(seq 1 120); do
  if curl -s -m 3 http://localhost:8041/v1/models >/dev/null 2>&1; then
    echo "prod up on OUR build: $(curl -s http://localhost:8041/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])')"
    exit 0
  fi
  sleep 10
done
echo "!!! lane did not come up -- restore the unit backup and daemon-reload"
exit 1
