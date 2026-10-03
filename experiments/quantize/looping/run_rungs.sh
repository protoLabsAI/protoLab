#!/usr/bin/env bash
# Sweep the quant ladder for looping: sampling arms x prompts, context depth,
# and thinking-on. One server per rung, killed by PID (never pkill -f).
set -u
SP=/tmp/claude-1001/-home-ava-dev-lab/60697cca-4e04-40f9-a218-548c0e1c5fb5/scratchpad
OUT=/mnt/data/gguf-forge/Ornith-1.5-9B-MTP/out
BIN=~/dev/llama.cpp/build-cuda/bin/llama-server
PORT=8099

serve() { # $1 = gguf path
  CUDA_VISIBLE_DEVICES=0 "$BIN" --model "$1" \
    --n-gpu-layers 99 -fit off --ctx-size 40960 --flash-attn on --jinja \
    --host 127.0.0.1 --port $PORT --no-webui > "$SP/server-cur.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    sleep 2
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0
  done
  echo "SERVER FAILED TO COME UP for $1"; tail -20 "$SP/server-cur.log"; return 1
}

stop() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; sleep 8; SRV=""; }

cd "$SP" || exit 1

for RUNG in Q4_K_M IQ4_XS IQ2_M; do
  F="$OUT/Ornith-1.5-9B-MTP-$RUNG.gguf"
  echo "=========== $RUNG ==========="
  serve "$F" || continue
  echo "--- sampling sweep ($RUNG) ---"
  python3 sweep.py $PORT "$RUNG" "$SP/results.jsonl"
  echo "--- depth ($RUNG) ---"
  python3 depth.py $PORT "$RUNG" "$SP/depth.jsonl" \
      4000,12000,20000,28000,36000 llamacpp_default,upstream_general
  echo "--- thinking-on ($RUNG) ---"
  THINK=1 MAXTOK=4096 python3 sweep.py $PORT "$RUNG-think" "$SP/think.jsonl" \
      llamacpp_default upstream_general
  stop
done
echo "ALL RUNGS DONE"
