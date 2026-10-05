#!/usr/bin/env bash
# Ornith-1.5-35B-A3B-NVFP4 smart lane — single GPU, :8041, served as smart/reasoning/coder.
#
# Arch: Qwen3_5MoeForConditionalGeneration — 35B total / 3B active, 256 experts (8/tok),
# 40 layers, native VL (images + video), 262K ctx. Registered by vllm-025 already; the
# Qwen3.5 family recipe applies unchanged.
#
# --- Two hard constraints on sm120, both learned the hard way ---
#
# 1) MoE NVFP4 REQUIRES --moe-backend marlin. The trtllm auto-backend segfaults on the
#    Sm120_SafeFP4 kernel even on clean checkpoints. Do not remove this flag.
#
# 2) marlin and MTP are MUTUALLY EXCLUSIVE (the global backend would also have to serve the
#    bf16 draft MoE), so this lane runs WITHOUT speculative decoding — SPEC_K=0. That is not
#    a loss worth chasing: MTP helps dense models and HURTS MoE (routing overhead exceeds the
#    speculation win; measured -11% on a 35B MoE). The checkpoint does ship 785 mtp.* tensors;
#    leave them unused.
#
# NOTE the speed numbers this lane was picked on were measured MTP-LESS and still beat the
# Qwen3.8-27B+MTP incumbent: decode C=1 266 vs 106 tok/s, TPOT C=8 6.7 vs 11.3 ms,
# TTFT @8k 270 vs 1148 ms. The incumbent is UNDERSTATED in that comparison because
# speed-test-v2's random dataset defeats its draft head — treat the margin as ~1.5-2x, not 2.5x.
#
# --generation-config auto is LOAD-BEARING, not boilerplate. It picks up the model's own
# generation_config.json (temperature 1.0 / top_p 0.95 / top_k 20). Ornith-1.5 FAILS TO
# TERMINATE at low temperature: at temp 0.2 it ran to a 32768-token cap emitting no code at
# all on hard problems. Do not pin a low temperature on this lane.
#
# Parsers: qwen3_xml + qwen3, both validated against the shipped chat_template.jinja
# (which is byte-identical to the base repo's — the NVFP4 quant did not mangle it).
# Measured with these ON UPSTREAM'S MODELOPT BUILD (not what this lane serves since
# 2026-08-21): claw 0.752, function_call 0.870 (identical to the incumbent's FC).
#
# Known regression vs Qwen3.8-27B: LiveCodeBench 0.365 vs 0.632. That is real and is the
# thing to watch on this lane. It is NOT a sampling artifact — raising temperature made it
# slightly worse (0.192 -> 0.163 thinking-off). Depth-coherence PASS to 255K (needle-exact),
# far beyond the incumbent's verified 60K. Vision PASS.
# ^ ALL OF THE ABOVE IS UPSTREAM'S BUILD. Ours is un-scorecarded — see the block below.
set -euo pipefail

# 2026-08-21: prod cut over from upstream's ModelOpt build to OUR compressed-tensors build.
#   ours     /mnt/models/quantized/Ornith-1.5-35B-A3B-NVFP4  25.0 GB  compressed-tensors
#            nvfp4-pack-quantized, ALL of linear_attn left bf16 (the DeltaNet finding)
#   upstream ornith-ai/Ornith-1.5-35B-A3B-NVFP4              23.4 GB  ModelOpt 0.45
#            MIXED_PRECISION + FP8 KV, and it DOES quantize linear_attn.out_proj
# Upstream's local copy was deleted in the same pass; re-download it if you need to A/B.
# Gate on our build: completion PASS, tool call PASS, vision 5/5, wordmark OCR 3/3 exact,
# needle-exact at 200,409 prompt tokens. No scorecard was run before the swap.
MODEL=${MODEL:-/mnt/models/quantized/Ornith-1.5-35B-A3B-NVFP4}
PORT=${PORT:-8041}
MAXLEN=${MAXLEN:-262144}
UTIL=${UTIL:-0.62}
MAXSEQS=${MAXSEQS:-16}
SERVED_NAMES=${SERVED_NAMES:-smart reasoning coder ornith-1.5-35b}
GPU=${GPU:-1}
TOOLP=${TOOLP:-qwen3_xml}
MOE_BACKEND=${MOE_BACKEND:-marlin}
SPEC_K=${SPEC_K:-0}          # see constraint (2) above — leave at 0

VENV=${VENV:-$HOME/dev/vllm-025}
CU13="$VENV/lib/python3.12/site-packages/nvidia/cu13"

# sm120 house recipe
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
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU

SPEC_ARGS=()
if [ "$SPEC_K" -gt 0 ] 2>/dev/null; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}")
fi

# shellcheck disable=SC2086
exec "$VENV/bin/vllm" serve "$MODEL" \
  "${SPEC_ARGS[@]}" \
  --served-model-name $SERVED_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization "$UTIL" \
  --moe-backend "$MOE_BACKEND" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOLP" \
  --reasoning-parser qwen3 \
  --generation-config auto \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --trust-remote-code \
  "$@"
