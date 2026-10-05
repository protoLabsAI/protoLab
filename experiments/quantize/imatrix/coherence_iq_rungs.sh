#!/usr/bin/env bash
# Coherence gate for the low-bit rungs. This is the one that matters at 2-3 bits:
# speed and acceptance both look fine on a checkpoint that has quietly gone to soup.
#
# NOTE the trap that produced three false "empty output" readings while building this:
# Ornith-1.5 thinks adaptively, so any budget under ~2K tokens returns EMPTY content and
# tells you nothing about coherence (feedback_eval_prod_token_budget).
set -uo pipefail
FORGE=/mnt/data/gguf-forge/Ornith-1.5-9B-MTP
BIN=/home/ava/dev/llama.cpp/build-cuda/bin/llama-server
PORT=8099
RUNGS=${RUNGS:-"IQ4_XS IQ3_M IQ2_M"}
OUT=$FORGE/imatrix/coherence

mkdir -p "$OUT"
for RUNG in $RUNGS; do
  GGUF="$FORGE/out/Ornith-1.5-9B-MTP-${RUNG}.gguf"
  [[ -f "$GGUF" ]] || { echo "skip $RUNG"; continue; }
  echo "=== $RUNG ==="
  CUDA_VISIBLE_DEVICES=0 "$BIN" --model "$GGUF" --n-gpu-layers 99 --ctx-size 40960 \
    --flash-attn on --jinja --port $PORT --host 127.0.0.1 \
    --spec-type draft-mtp --spec-draft-n-max 3 > "$OUT/$RUNG-server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 2; done

  /home/ava/dev/lab/.venv/bin/python /home/ava/dev/lab/evals/graders/verify_coherence.py \
    --base-url http://127.0.0.1:$PORT/v1 --model "$RUNG" \
    --depths 4096,16384,32768 > "$OUT/$RUNG-coherence.txt" 2>&1
  echo "  verify_coherence exit=$?"
  tail -6 "$OUT/$RUNG-coherence.txt"

  python3 - "$RUNG" "$OUT" <<'PY'
import json, sys, urllib.request
rung, out = sys.argv[1], sys.argv[2]
PROMPTS = [
  ("prose","Write an opening paragraph for a story about a lighthouse keeper who stops receiving mail."),
  ("math","A train leaves at 14:20 travelling 80 km/h. A second leaves the same station at 15:05 travelling 110 km/h on the same track. When does the second catch the first? Show your working."),
  ("code","Write a Python function that merges two sorted lists into one sorted list, then state its time complexity."),
]
res={}
for tag,p in PROMPTS:
    body=json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":4096,"temperature":0.7}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:8099/v1/chat/completions",data=body,
        headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req,timeout=600))
    m=d["choices"][0]["message"]
    res[tag]=(m.get("content") or "").strip() or (m.get("reasoning") or "").strip()
    print(f"  [{tag}] {len(res[tag])} chars: {res[tag][:150]!r}")
json.dump(res, open(f"{out}/{rung}-samples.json","w"), indent=2)
PY
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3
done
