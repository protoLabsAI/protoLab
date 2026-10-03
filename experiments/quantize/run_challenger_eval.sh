#!/bin/bash
# End-to-end, UNATTENDED challenger eval:
#   wait for download → stop coder lane → NVFP4 quantize → serve → eval-model.sh --full
#   → LCB thinking-on probe → restore coder lane → print comparison.
# Safe: a trap restores the coder lane on ANY exit. Aborts (and restores) if a phase fails.
set -uo pipefail

SRC=/mnt/data/models-src/Coder-Opus-Distill-bf16
NVFP4=/mnt/data/models/Coder-Opus-Distill-NVFP4
PORT=8033
SERVED=challenger
LABEL=Opus-Distill
EVAL_DIR=/home/ava/dev/lab/evals
LOG=/mnt/data/challenger-eval/pipeline.log
SERVE_LOG=/mnt/data/challenger-eval/serve.log
QUANT_LOG=/mnt/data/challenger-eval/quantize.log
mkdir -p /mnt/data/challenger-eval /mnt/data/.offload-challenger
exec > >(tee -a "$LOG") 2>&1

SERVE_PID=""
say(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }
restore_coder(){
  say "RESTORE: stopping challenger serve + restarting coder lane + Fish + embed"
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
  # free any lingering challenger vLLM on the port, then bring the GPU1 stack back
  sleep 5
  sudo systemctl start vllm-coder 2>/dev/null
  sudo systemctl start protovoice-stack embed-server 2>/dev/null
  say "GPU1 stack restart issued (coder + protovoice + embed)"
}
trap restore_coder EXIT

say "challenger pipeline start"

# ── 0. ensure download complete — just RESUME directly (hf download is fast + resumes; the
# earlier "throttle" was a bogus tiny-file speed probe, NOT a real limit). Retry on failure.
say "PHASE 0: ensuring download complete (direct resume, retry on failure)"
TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
tries=0
while true; do
  n=$(ls "$SRC"/*.safetensors 2>/dev/null | wc -l)
  realsz=$(du -cb "$SRC"/*.safetensors 2>/dev/null | tail -1 | cut -f1); realsz=${realsz:-0}
  if [ "$n" -ge 4 ] && [ "$realsz" -ge 150000000000 ]; then
    say "download complete: $n shards, $(( realsz/1000000000 ))G real"; break
  fi
  tries=$((tries+1))
  say "download attempt #$tries (have $n/4, $(( realsz/1000000000 ))G)"
  env HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN="$TOKEN" HF_HOME=/mnt/data/.hf-cache-challenger \
    /home/ava/dev/vllm-env/bin/hf download samuelcardillo/Qwen3-Coder-Next-Opus-4.6-Reasoning-Distilled \
    --local-dir "$SRC" >> /mnt/data/.hf-cache-challenger/download.log 2>&1
  say "attempt #$tries exited $? (now $(ls "$SRC"/*.safetensors 2>/dev/null | wc -l)/4)"
  [ "$tries" -ge 20 ] && { say "ABORT: 20 download attempts failed"; exit 1; }
  sleep 10
done

# ── 1. stop coder lane (free GPU1) ──
say "PHASE 1: freeing GPU1 (stop coder + Fish + embed — quantize needs the whole card)"
sudo systemctl stop vllm-coder protovoice-stack embed-server
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

# ── 2. NVFP4 quantize (GPU1 now fully free, disk-offload for the tail) ──
say "PHASE 2: NVFP4 quantize (this is the long pole, ~1-2h)"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/dev/quant-env/bin/python \
  /home/ava/dev/lab/experiments/quantize/coder_opus_distill_requant.py > "$QUANT_LOG" 2>&1
if [ ! -f "$NVFP4/config.json" ]; then
  say "ABORT: quantize did not produce $NVFP4/config.json — see $QUANT_LOG"; tail -20 "$QUANT_LOG"; exit 1
fi
say "quantize done: $(du -sh "$NVFP4" | cut -f1)"

# ── 3. serve challenger on vLLM (GPU1), sm120 recipe env ──
say "PHASE 3: serving challenger NVFP4 on :$PORT"
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/mnt/models/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_TRITON_FP8_GEMM=1 \
CUDA_HOME=/home/ava/dev/vllm-024-test/lib/python3.12/site-packages/nvidia/cu13 \
PATH=/home/ava/dev/vllm-024-test/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
FLASHINFER_CUDA_ARCH_LIST=12.0f NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK MAX_JOBS=4 \
/home/ava/dev/vllm-024-test/bin/vllm serve "$NVFP4" \
  --host 0.0.0.0 --port $PORT --served-model-name $SERVED \
  --max-model-len 262144 --max-num-seqs 8 --gpu-memory-utilization 0.60 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --moe-backend marlin --kv-cache-dtype fp8 --enable-chunked-prefill --enable-prefix-caching \
  --trust-remote-code > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
say "serve pid $SERVE_PID — waiting for health (up to 20 min: cutlass/marlin JIT)"
t=0
until curl -s http://127.0.0.1:$PORT/health -o /dev/null -w '%{http_code}' --max-time 4 2>/dev/null | grep -q 200; do
  sleep 20; t=$((t+20))
  kill -0 "$SERVE_PID" 2>/dev/null || { say "ABORT: serve process died — see $SERVE_LOG"; tail -25 "$SERVE_LOG"; exit 1; }
  [ "$t" -ge 1200 ] && { say "ABORT: serve health timeout"; tail -25 "$SERVE_LOG"; exit 1; }
done
say "challenger serving on :$PORT"

# quick coherence + capability sanity
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with only: OK\"}],\"max_tokens\":800}" \
  --max-time 60 | ~/dev/vllm-env/bin/python -c "import sys,json;m=json.load(sys.stdin)['choices'][0]['message'];print('  sanity:',repr((m.get('content') or m.get('reasoning_content') or '')[:60]))" 2>/dev/null

# ── 4. full scorecard (LCB thinking-off = capability, as eval-model.sh does) ──
say "PHASE 4: eval-model.sh --full (LCB thinking-off capability)"
cd "$EVAL_DIR"
./eval-model.sh "$LABEL" "http://localhost:$PORT/v1" "$SERVED" --full

# ── 5. LCB thinking-ON probe (does the Opus reasoning converge on code or run away?) ──
say "PHASE 5: LCB thinking-ON probe (30 problems, prod-representative)"
SCD=$(ls -td "$EVAL_DIR"/results/scorecard-$LABEL-*/ 2>/dev/null | head -1)
set -a; source "$EVAL_DIR/.env"; set +a
export JUDGE_GATEWAY_URL=http://ava:4000/v1 JUDGE_MODEL=protolabs/cloud
"$EVAL_DIR/venv/bin/python" -m runners.run_livecodebench --model "$SERVED" \
  --gateway-url "http://localhost:$PORT/v1" --limit 30 --min-date 2025-01-01 --max-tokens 32768 \
  --output-dir "${SCD}lcb-thinking-on" > "${SCD}lcb-thinking-on.log" 2>&1
say "thinking-on LCB: $(grep -E 'Mean score|no-code-in-budget' "${SCD}lcb-thinking-on.log" | tail -2 | tr '\n' ' ')"

say "challenger pipeline COMPLETE — scorecard: ${SCD}scorecard.md"
echo "DONE_MARKER_CHALLENGER"
# trap restores the coder lane on exit
