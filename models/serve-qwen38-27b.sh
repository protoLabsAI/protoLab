#!/usr/bin/env bash
# Qwen3.8-27B bf16 eval serve — single GPU 0, :8000, served as `local`.
#
# Qwen3.8 is a NEW POST-TRAIN ON THE QWEN3.5 ARCHITECTURE, not a new arch:
# config.json declares model_type=qwen3_5 / Qwen3_5ForConditionalGeneration, which both
# vllm-025 (0.25.1) and vllm-jasl already register — no bump, no fork needed.
#   dense 27B · 64 layers · 16 x (3 x GatedDeltaNet->FFN + 1 x GatedAttention->FFN)
#   native VL (27-layer ViT, images + video) · MTP head (mtp_num_hidden_layers=1) · 262K ctx
#
# Parsers verified against the shipped chat_template.jinja, NOT guessed: tool calls are the
# XML form (<tool_call><function=name><parameter=k>v</parameter></function></tool_call>) and
# thinking is plain <think>...</think> -> qwen3_xml + qwen3. The classic `hermes` JSON-in-
# <tool_call> parser will NOT parse this template.
#
# Thinking is ON by default with preserve_thinking=true. That is a real token-cost lever, not a
# nuisance knob (see feedback_eval_prod_token_budget) — eval at the budget prod would actually
# use, and record which effort level produced the numbers.
#
# EFFORT DEFAULT IS PINNED TO `medium` VIA A CHAT-TEMPLATE OVERRIDE (2026-09-08).
# The model's shipped chat_template.jinja does `reasoning_effort|default('xhigh')`. A
# 252-generation judge-free sweep (experiments/effort-sweep/, run-20260908-061817) found xhigh
# is NOT deeper thinking — it is a TERMINATION failure. Conditioned on finishing it is the best
# arm on both suites (0.932 / 1.000), but it burns the full 32k budget and scores 0 on 19% of
# reasoning_hard and 6% of lcb_medium runs; on LiveCodeBench *hard* it failed to return at all
# on 12 of 18 item-matched runs even at a 3600 s client timeout. `medium` ties or beats every
# arm at 53–80% fewer reasoning tokens and zero cap-outs.
#
# The gateway already pins low/medium on every alias, so this default only ever reached callers
# hitting :8041/:8042 DIRECTLY (eval scripts, tailnet consumers that skip the proxy) — exactly
# the traffic that had no protection. See protoLabsAI/homelab-iac#255.
# Set CHAT_TEMPLATE= (empty) to fall back to the model's shipped template.
#
# --enable-prompt-tokens-details is REQUIRED for anyone to SEE prefix caching. Without it
# usage.prompt_tokens_details comes back null on every response even when the lane is hitting
# ~72% cache on warm prefixes (measured 2026-08-17, protoLab#28) -- the gateway and every
# other consumer then read a hard zero forever and cannot tell a cache regression from a load
# spike. The flag was broken on the V1 engine for 14+ months (vllm#16162/#18062/#44961) and
# is fixed in 0.23-0.25.1, which is what this lane runs. It is reporting only, no perf cost.
#
# PREREQ: vllm-smart.service (DSV4 TP=2) must be stopped — it holds both cards.
# MTP is intentionally OFF here: it is lossless, so it moves speed only, and leaving it off
# keeps the quality baseline free of spec-decode confounds. Turn it on for the speed pass.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-262144}
UTIL=${UTIL:-0.90}
MAXSEQS=${MAXSEQS:-16}
SERVED_NAMES=${SERVED_NAMES:-local qwen3.8-27b}
GPU=${GPU:-0}
# MTP draft head ships as a bf16 sidecar (model-mtp.safetensors) in the NVFP4 artifact.
# The JSON is BUILT HERE from scalars, never passed as a quoted env value: systemd strips
# double quotes from Environment= and ExecStart=, which turns the config into
# {method:mtp,num_speculative_tokens:1} and vLLM rejects it at argparse (json.loads fails).
# Tune with SPEC_K / SPEC_METHOD; set SPEC_K=0 to serve without speculative decoding.
# Tool parser. `qwen3_xml` was read off the shipped chat_template (XML <function=…> form);
# the official vLLM recipe for this model specifies `qwen3_coder`. Both resolve to
# qwen3_engine_tool_parser.py — A/B them before assuming either is right (FC is the one axis
# where this lane trails DSV4: 0.870 vs 0.907).
TOOLP=${TOOLP:-qwen3_coder}
SPEC_METHOD=${SPEC_METHOD:-mtp}
SPEC_K=${SPEC_K:-1}
if [ "$SPEC_K" -gt 0 ] 2>/dev/null; then
  SPEC=${SPEC:-"{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_K}"}
else
  SPEC=""
fi

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
[ -n "$SPEC" ] && SPEC_ARGS=(--speculative-config "$SPEC")

# Effort-default override (see the header note). Differs from the model's shipped template by
# exactly one line: reasoning_effort|default('xhigh') -> default('medium').
CHAT_TEMPLATE=${CHAT_TEMPLATE-"$(cd "$(dirname "$0")" && pwd)/chat-templates/qwen38-effort-medium.jinja"}
TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE" ]; then
  if [ ! -f "$CHAT_TEMPLATE" ]; then
    echo "FATAL: CHAT_TEMPLATE not found: $CHAT_TEMPLATE" >&2
    echo "       Serving without it would silently restore the xhigh default." >&2
    exit 1
  fi
  # Staleness guard. This override is a COPY of one model's chat template, so a model bump can
  # silently pin an outdated tool-call/thinking format. Compare against the shipped template
  # ignoring the one line we intend to differ; warn loudly rather than refuse to start.
  SHIPPED="$MODEL/chat_template.jinja"
  if [ -f "$SHIPPED" ]; then
    if ! diff -q <(grep -v "reasoning_effort|default" "$SHIPPED") \
                 <(grep -v "reasoning_effort|default" "$CHAT_TEMPLATE") >/dev/null; then
      echo "WARNING: $CHAT_TEMPLATE differs from $SHIPPED by more than the effort default." >&2
      echo "         Regenerate it from the shipped template before trusting this lane." >&2
    fi
  fi
  TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
fi

# shellcheck disable=SC2086
exec "$VENV/bin/vllm" serve "$MODEL" \
  "${SPEC_ARGS[@]}" \
  --served-model-name $SERVED_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization "$UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOLP" \
  --reasoning-parser qwen3 \
  "${TEMPLATE_ARGS[@]}" \
  --generation-config auto \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --trust-remote-code \
  "$@"
