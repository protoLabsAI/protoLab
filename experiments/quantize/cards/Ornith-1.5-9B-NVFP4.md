---
license: mit
base_model: ornith-ai/Ornith-1.5-9B
base_model_relation: quantized
tags:
  - nvfp4
  - vllm
  - compressed-tensors
  - blackwell
  - mtp
  - speculative-decoding
  - vision
pipeline_tag: image-text-to-text
---

# Ornith-1.5-9B — NVFP4 (vLLM, Blackwell)

W4A4 NVFP4 quant of [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B)
for vLLM on Blackwell (sm120), **with a distilled MTP draft head included**. Upstream ships
1.5-9B as bf16, GGUF and MLX — there is no NVFP4 build anywhere else.

Built because [someone asked for one](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-MTP-GGUF/discussions/1).

- **11.2 GB** for the quantized model, down from 17.9 GB bf16.
- **Vision intact and verified against the bf16 source**, not just "it returned something".
- **MTP head ships with it** (`model-mtp.safetensors`, 15 tensors, bf16) — Ornith-1.5-9B has
  *none* upstream, so this is our own KL-distilled head, the same one in
  [`Ornith-1.5-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-MTP-GGUF).

## What is and isn't quantized

| Component | Precision | Why |
|---|---|---|
| LM attention + MLP linears (128) | **NVFP4 W4A4** | the win |
| Vision tower (333 tensors) | bf16 | no sm120 W4A4 kernel for it |
| DeltaNet / GDN `linear_attn` | bf16 | low-precision activations corrupt DeltaNet — standing finding on this arch |
| `lm_head`, `embed_tokens` | bf16 | quantizing `lm_head` is the known vLLM NVFP4 crash |
| MTP head (`mtp.*`) | bf16 | drafts only; the target verifies every token |

## Run

```bash
vllm serve protoLabsAI/Ornith-1.5-9B-NVFP4 \
  --max-model-len 32768 --gpu-memory-utilization 0.30 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 --generation-config auto --trust-remote-code
```

`--generation-config auto` is **load-bearing, not boilerplate.** It picks up the model's own
sampling defaults. The Ornith-1.5 family fails to terminate at low temperature — pinning a low
temp will run it to your token cap producing nothing useful.

On sm120 also set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_TRITON_FP8_GEMM=1`.

**Budget your tokens.** Ornith-1.5 thinks adaptively, and a short cap returns an EMPTY
completion — all of it went to reasoning. We tripped this three separate times building this
release: at `max_tokens=400` the model returned 0 characters and `finish_reason=length`; the
same prompt at 4096 returned a clean 724-character answer plus a 5046-character trace. If you
get blank responses, raise the budget before suspecting the weights.

## Speed

Concurrency-swept (`vllm bench serve`, random dataset, cache-cold), RTX PRO 6000 Blackwell,
co-tenant lanes live on the box — so these are honest-but-not-quiet-GPU numbers:

    regime          C   ttft p50   tpot p50   agg tok/s   goodput
    chat 1k/1k      1       63ms      6.7ms       148.5      0.15
    chat 1k/1k      8      296ms      7.3ms      1059.0      1.03
    context 8k/1k   1      305ms      6.8ms       140.2      0.14
    context 8k/1k   8     1297ms      9.2ms       775.4      0.69

**MTP is worth turning on: 1.27x.** Same three coherent prompts, greedy, same lane, only
`--speculative-config` changed:

    arm                 decode      acceptance
    MTP off             151.1 t/s        —
    MTP on (K=1)        192.5 t/s      0.804

```bash
vllm serve protoLabsAI/Ornith-1.5-9B-NVFP4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' ...
```

vLLM resolves the architecture to `Qwen3_5MTP` and shares the target's embedding and `lm_head`
with the drafter — no separate draft model to wire up.

Note the **0.804 acceptance is on real prompts**. Benchmarks that feed random tokens report far
lower acceptance for any speculative decoder, because a draft head cannot predict noise — if
you measure this lane with `--dataset-name random` you are measuring the dataset.

## Vision: measured against the source, not asserted

A quantized VL checkpoint that loads fine and serves text perfectly can have a completely dead
vision path — so "we ran an image through it" is not evidence. What matters is the *difference*
from the source. Both served identically, n=20 per side, temperature 0.7:

    probe                          bf16      NVFP4    p (Fisher, 2-sided)
    ---------------------------    -------   -------  -------------------
    shapes (red circle/blue sq)    20/20     20/20    1.00
    wordmark OCR, exact             1/20      1/20    1.00
    wordmark OCR, token correct     7/20     13/20    0.11

**No detectable loss.** Note the wordmark row: *both* precisions score 1/20 exact. The bf16
model itself misreads the stylised "protoLabs" as "protocolabs" and invents a trailing digit
("VLM-429"). That is a base-model weakness on a hard glyph, not quantization damage — and it is
exactly why the gate scores the difference rather than an absolute threshold. An earlier n=5
read showed bf16 2/5 vs NVFP4 0/5 and looked like real damage; at n=20 it vanished.

## Scorecard

Discriminating frontier battery against this build. Judge-free except claw, which uses an
independent cloud judge so a local model never grades itself:

    axis            score   kind                detail
    --------------  -----   ------------------  ------
    function_call   0.963   schema-checked      52/54 · untagged 100% · in-proc 100% · ext 90%
    claw            0.675   agentic/LLM-judged  10 tasks · robustness 1.00 · safety-clean
    reasoning_hard  0.611   solver-verified     5/9 full-pass
    livecodebench   0.115   exec-graded         hard-only, 30 problems, thinking-off

Judge reported **0 fallbacks**, so the LLM-judged score is real rather than a dead judge
defaulting to 0.5.

**function_call 0.963 is the best result on our internal board** — across ~26 scorecards
spanning 9B to 397B, including dedicated coder models. For a 9B that is the reason to run
this model: schema-correct tool calls, 100% on both the untagged and in-process suites.

**LiveCodeBench 0.115 is a real weakness, and it is the model, not the quantization.**
12 of 30 problems earned partial credit; none passed every test. The mechanism is budget
exhaustion — **13 of 30 problems consumed the entire 32,768-token budget** deliberating and
never emitted working code. This matches what users independently report about the Ornith-1.5
family (failed one-shot HTML tasks, regressions versus Ornith-1.0, context exhaustion), and
we measured the same signature on the 35B. Thinking-off does not rescue it: on the 35B we
paired thinking-on against thinking-off on identical problems and thinking-on was *worse*
(0.129 vs 0.329) while exhausting the budget on 6 of 7.

**If code generation is your workload, this family is not the right pick at any precision.**
If tool calling is, it is excellent.

## Release gate

    completion   PASS   coherent, correct, terminates
    tool call    PASS   qwen3_xml, correct name + parsed arguments
    vision       PASS   5/5 shapes; parity vs bf16 at n=20 (table above)
    census       PASS   128 LM linears packed; visual/linear_attn/mtp/lm_head packed = 0
                        MTP sidecar present, 15 tensors; no key-prefix mangling

## Provenance & license

- **Base:** `ornith-ai/Ornith-1.5-9B` (MIT) — dense Qwen3.5-9B-architecture hybrid (linear +
  full attention) VL fine-tune, trained with end-to-end RL self-improvement.
- **MTP head:** grafted from `Qwen/Qwen3.5-9B` (Apache-2.0), then KL-distilled against
  Ornith-1.5-9B's own hidden states.
- Quantized with llm-compressor (compressed-tensors NVFP4). **MIT.**
  Built by [protoLabs.studio](https://protolabs.studio).
