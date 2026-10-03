#!/usr/bin/env bash
# Test lane for protoLab#36: replica B's exact unit config (vllm-smart-qwen38-b.service +
# its drop-ins) on GPU0, but on :8060 so the gateway never routes prod traffic here.
# Usage: SPEC_K=0|1|3 ./lane.sh   (replica B must be stopped first; GPU0 can't hold both)
set -euo pipefail
export MODEL=/mnt/models/quantized/ukisai-Swift-Qwen3.8-27B-NVFP4
export GPU=0 PORT=8060 MAXLEN=262144 UTIL=0.86 MAXSEQS=32
export SERVED_NAMES="smart"
export SPEC_K=${SPEC_K:?set SPEC_K}
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
# OVERLAY=<dir> puts a patched copy of the vllm package ahead of site-packages (prod env untouched)
[ -n "${OVERLAY:-}" ] && export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
exec /home/ava/dev/lab/models/serve-qwen38-27b.sh >> "/mnt/scratch/logs/longctx-eos-k${SPEC_K}${OVERLAY:+-patched}.log" 2>&1
