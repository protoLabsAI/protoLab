---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - nvfp4
  - vllm
  - compressed-tensors
  - blackwell
  - moe
  - vision
pipeline_tag: image-text-to-text
---

# Ornith-1.5-35B-A3B — NVFP4 (compressed-tensors, vLLM/Blackwell)

W4A4 NVFP4 quant of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
— 35B total / 3B active, 256 experts (8/tok), 40 layers, native VL.

**25.0 GB.** This is the build we run as our own smart lane in production.

> **Upstream also ships an NVFP4** at [`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4),
> and it is good — we ran it in prod for a day. This is not a "first" and does not claim to
> beat it. It exists because [someone using our 1.0 NVFP4 asked for the 1.5 equivalent](https://huggingface.co/protoLabsAI/Ornith-1.0-35B-NVFP4/discussions/1),
> and because it is a **different quantization**, not a repackage:

| | this build | upstream |
|---|---|---|
| framework | llm-compressor / **compressed-tensors** | NVIDIA **ModelOpt** 0.45 |
| format | `nvfp4-pack-quantized` | `MIXED_PRECISION` |
| KV cache | not pinned (serve as you like) | FP8 |
| DeltaNet (`linear_attn`) | **entirely bf16** | `out_proj` quantized |
| size | 25.0 GB | 23.4 GB |

The DeltaNet choice is the reason for the 1.6 GB difference. Low-precision activations
corrupt DeltaNet on this architecture — a standing finding across our Qwen3.5-family quants —
so we keep the whole linear-attention path out of it. Whether that matters for *your*
workload is an open question; upstream's build works. Pick on measurements, not vibes.

## What is and isn't quantized

| Component | Precision |
|---|---|
| 30,720 expert projections (40 × 256 × 3) | **NVFP4 W4A4** |
| 160 attention projections | **NVFP4 W4A4** |
| Router (`mlp.gate`), `shared_expert_gate` | bf16 — low-precision routing corrupts MoE |
| DeltaNet / GDN `linear_attn` | bf16 |
| Vision tower | bf16 |
| `lm_head`, `embed_tokens` | bf16 |
| MTP head (785 `mtp.*` tensors) | bf16, **shipped but not servable here** — see below |

## Run

```bash
vllm serve protoLabsAI/Ornith-1.5-35B-A3B-NVFP4 \
  --moe-backend marlin \
  --max-model-len 262144 --gpu-memory-utilization 0.62 --max-num-seqs 16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --generation-config auto --trust-remote-code
```

Three flags are load-bearing, all learned the hard way on sm120:

- **`--moe-backend marlin` is required.** The trtllm auto-backend segfaults on the
  `Sm120_SafeFP4` kernel even with a clean checkpoint. Don't remove it.
- **`--generation-config auto` is not boilerplate.** The Ornith-1.5 family fails to terminate
  at low temperature — pin a low temp and it runs to your token cap emitting nothing useful.
- **Budget your tokens.** It thinks adaptively; a short `max_tokens` returns *empty content*
  with `finish_reason=length` because the whole budget went to the reasoning channel. If you
  get blank replies, raise the budget before suspecting the weights.

**MTP:** the 785 `mtp.*` tensors are preserved, but marlin and MTP are mutually exclusive
(the global MoE backend would also have to serve the unquantized bf16 draft MoE), so this
lane runs without speculative decoding on vLLM/sm120 today. Shipped anyway — other backends
may take them. MTP also *hurts* MoE in our measurements (routing overhead exceeds the
speculation win), so this is not the loss it sounds like.

## Verification

    census        30,720 expert + 160 attn packed W4A4; ZERO packed in
                  visual / linear_attn / mtp / lm_head / router; 785 mtp.* preserved
    completion    PASS — coherent, correct, terminates
    tool call     PASS — qwen3_xml, correct name + parsed arguments
    vision        PASS — 5/5 shapes; wordmark OCR 3/3 exact
    depth         needle-exact at 200,409 prompt tokens
    coherence     clean detectors at 32K and 131K

## Scorecard

Discriminating frontier battery, run against **this build serving in production**, judge-free
except claw (independent cloud judge — a local model never grades itself):

    axis            score   kind                detail
    --------------  -----   ------------------  ------
    claw            0.719   agentic/LLM-judged  10 tasks · robustness 1.00 · safety-clean
    reasoning_hard  0.861   solver-verified     7/9 full-pass
    function_call   0.889   schema-checked      48/54 · untagged 100% · in-proc 85% · ext 90%
    livecodebench   0.205   exec-graded         hard-only, 30 problems, thinking-off

Judge reported **0 fallbacks**, so the LLM-judged score is real and not a dead judge
defaulting to 0.5.

**On LiveCodeBench — read this before drawing a conclusion.** 0.205 is partial credit
(per-test pass rate); zero of the 30 *hard* problems passed every test. Code generation was
not broken: individual problems scored up to 0.95 with 17/20 tests passing, no errors, no
truncation.

LiveCodeBench is weak across the whole Ornith-1.5-35B family, not just this quant. We ran the
**upstream ModelOpt build through the identical harness** and it scored **0.192** — this build
scores 0.205, i.e. no quantization penalty, if anything a hair above. Independent users report
the same coding weakness on the unquantized model (see the upstream repo discussions: failed
one-shot HTML tasks, regressions versus Ornith-1.0, context exhaustion from over-deliberation).
So: the number is real, it is a property of the model, and it is **not** caused by this quant.
If code generation is your workload, this family is not the right pick at any precision.

reasoning_hard 0.861 and function_call 0.889 are strong, and function_call is slightly ahead
of the upstream build's 0.870.

## Provenance & license

- **Base:** `ornith-ai/Ornith-1.5-35B-A3B` (MIT).
- Quantized with llm-compressor (compressed-tensors NVFP4), 128 calibration samples @2048
  from `ultrachat_200k`, `moe_calibrate_all_experts=True`. **MIT.**
  Built by [protoLabs.studio](https://protolabs.studio).
