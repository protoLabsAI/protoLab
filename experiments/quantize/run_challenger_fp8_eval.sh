#!/bin/bash
# Serve the streaming-FP8 challenger + run the full scorecard. Quantize already done (76G FP8).
# Frees GPU1 (coder+Fish+embed), serves block-wise FP8 on sm120, evals, restores. Trap-safe.
set -uo pipefail
FP8=/mnt/data/models/Coder-Opus-Distill-FP8
PORT=8033; SERVED=challenger; LABEL=Opus-Distill
EVAL_DIR=/home/ava/dev/lab/evals
LOG=/mnt/data/challenger-eval/fp8-eval.log
SERVE_LOG=/mnt/data/challenger-eval/fp8-serve.log
exec > >(tee -a "$LOG") 2>&1
SERVE_PID=""
say(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }
restore(){
  say "RESTORE: stop challenger serve + restart GPU1 stack (coder+Fish+embed)"
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null
  sleep 5
  sudo systemctl start vllm-coder protovoice-stack embed-server 2>/dev/null
  say "GPU1 stack restart issued"
}
trap restore EXIT
say "challenger FP8 eval start"

# 1. free GPU1
say "PHASE 1: freeing GPU1 (stop coder+Fish+embed) for the 76G FP8"
sudo systemctl stop vllm-coder protovoice-stack embed-server
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

# 2. serve block-wise FP8 on sm120 (Triton FP8 path; NO --moe-backend marlin — that's NVFP4-only)
say "PHASE 2: serving challenger FP8 on :$PORT"
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/mnt/models/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_TRITON_FP8_GEMM=1 \
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
CUDA_HOME=/home/ava/dev/vllm-024-test/lib/python3.12/site-packages/nvidia/cu13 \
PATH=/home/ava/dev/vllm-024-test/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
FLASHINFER_CUDA_ARCH_LIST=12.0f NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK MAX_JOBS=4 \
/home/ava/dev/vllm-024-test/bin/vllm serve "$FP8" \
  --host 0.0.0.0 --port $PORT --served-model-name $SERVED \
  --max-model-len 131072 --max-num-seqs 8 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --kv-cache-dtype fp8 --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
  > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
say "serve pid $SERVE_PID — waiting for health (up to 20 min JIT)"
t=0
until curl -s http://127.0.0.1:$PORT/health -o /dev/null -w '%{http_code}' --max-time 4 2>/dev/null | grep -q 200; do
  sleep 20; t=$((t+20))
  kill -0 "$SERVE_PID" 2>/dev/null || { say "ABORT: serve died — see $SERVE_LOG"; tail -30 "$SERVE_LOG"; exit 1; }
  [ "$t" -ge 1200 ] && { say "ABORT: serve health timeout"; tail -30 "$SERVE_LOG"; exit 1; }
done
say "challenger serving on :$PORT"
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with only: OK\"}],\"max_tokens\":600}" \
  --max-time 60 | ~/dev/vllm-env/bin/python -c "import sys,json;m=json.load(sys.stdin)['choices'][0]['message'];print('  sanity:',repr((m.get('content') or m.get('reasoning_content') or '')[:80]))" 2>/dev/null

# 3. full scorecard (LCB thinking-off capability, as eval-model.sh does)
say "PHASE 3: eval-model.sh --full"
cd "$EVAL_DIR"
./eval-model.sh "$LABEL" "http://localhost:$PORT/v1" "$SERVED" --full

# 4. LCB thinking-ON probe (does the Opus reasoning converge on code?)
say "PHASE 4: LCB thinking-ON probe (30 problems)"
SCD=$(ls -td "$EVAL_DIR"/results/scorecard-$LABEL-*/ 2>/dev/null | head -1)
set -a; source "$EVAL_DIR/.env"; set +a
export JUDGE_GATEWAY_URL=http://ava:4000/v1 JUDGE_MODEL=protolabs/cloud
"$EVAL_DIR/venv/bin/python" -m runners.run_livecodebench --model "$SERVED" \
  --gateway-url "http://localhost:$PORT/v1" --limit 30 --min-date 2025-01-01 --max-tokens 32768 \
  --output-dir "${SCD}lcb-thinking-on" > "${SCD}lcb-thinking-on.log" 2>&1
say "thinking-on LCB: $(grep -E 'Mean score|no-code-in-budget' "${SCD}lcb-thinking-on.log" | tail -2 | tr '\n' ' ')"

say "challenger FP8 eval COMPLETE — scorecard: ${SCD}scorecard.md"
echo "DONE_MARKER_CHALLENGER_FP8"
