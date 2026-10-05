#!/usr/bin/env bash
# Ornith-1.5-9B-NVFP4 — dense hybrid-GDN VL lane. Gate/serve helper.
#
# Dense, so unlike the 35B-A3B MoE lane this one CAN run MTP: the marlin-vs-MTP conflict is
# a MoE-only constraint (the global moe-backend would have to serve the bf16 draft MoE).
# Set SPEC_K=1 to enable the distilled head shipped in this checkpoint (model-mtp.safetensors).
#
# --generation-config auto is load-bearing on the Ornith-1.5 family: it fails to terminate at
# low temperature (measured on the 35B — ran to a 32768-token cap emitting nothing).
set -euo pipefail

MODEL=${MODEL:-/mnt/models/quantized/Ornith-1.5-9B-NVFP4}
PORT=${PORT:-8062}
MAXLEN=${MAXLEN:-32768}
UTIL=${UTIL:-0.30}
MAXSEQS=${MAXSEQS:-8}
SERVED_NAMES=${SERVED_NAMES:-ornith-1.5-9b-nvfp4}
GPU=${GPU:-0}
TOOLP=${TOOLP:-qwen3_xml}
SPEC_K=${SPEC_K:-0}

VENV=${VENV:-$HOME/dev/vllm-025}
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"

export CUDA_HOME="$CU13"
export PATH="$CU13/bin:$PATH"
export LD_LIBRARY_PATH="$CU13/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export MAX_JOBS=4
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_TRITON_FP8_GEMM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/models/huggingface
export CUDA_VISIBLE_DEVICES=$GPU

SPEC_ARGS=()
if [ "$SPEC_K" -gt 0 ] 2>/dev/null; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}")
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  "${SPEC_ARGS[@]}" \
  --served-model-name $SERVED_NAMES \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization "$UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOLP" \
  --reasoning-parser qwen3 \
  --generation-config auto \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  "$@"
