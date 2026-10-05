---
license: mit
base_model: ornith-ai/Ornith-1.5-9B
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - mtp
  - multi-token-prediction
  - qwen3.5
  - vision
pipeline_tag: image-text-to-text
---

# Ornith-1.5-9B MTP — GGUF (llama.cpp speculative decoding)

GGUF builds of [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B) with a
**distilled MTP draft head baked into the trunk** — llama.cpp does lossless multi-token
self-speculative decoding out of the box, no separate draft model to wire up. Every file here
carries the `nextn` head, so `--spec-type draft-mtp` just works.

Ornith ships 1.5-9B with `mtp_num_hidden_layers: 1` in `config.json` but **none of the `mtp.*`
weights** — 0 of 760 tensors. So the stock GGUFs (official and third-party) have no MTP head and
can't speculate. These do.

- **Blackwell (RTX 50xx / PRO 6000): use `NVFP4`.** 6.5 GB, and the fastest rung here —
  **299 tok/s with MTP (1.38×)**. MTP's verify step is nearly free on FP4 tensor cores and
  costs real time on the K-quant dequant path, so the two compound.
- **6 GB card? Use `IQ4_XS`** (5.45 GB). It is smaller *than* `Q4_K_M`, faster, and takes a
  bigger MTP gain (1.21× vs 1.06×).
- **⚠️ `IQ2_M` degenerates on long generations — do not use it for prose or code.** It repeats
  itself in 28% of 1500-token generations at llama.cpp's default sampling. `IQ3_M` is marginal
  (7.8% at defaults, clean with the recommended sampler). Measured table below.
- **Use `Q8_0` for reference quality.** It takes the largest *relative* MTP gain, because
  its baseline is the most bandwidth-bound (1.57×–1.77× depending on prompt mix) — but it is
  still slower in absolute terms than `NVFP4`.

> **`Q4_K_M` is no longer the low-VRAM recommendation.** An earlier version of this card said it
> was. `IQ4_XS` beats it on size, speed and MTP gain — measured, table below.

> Want the base with no MTP head? `ornith-ai/Ornith-1.5-9B-GGUF`.

## Files

| File | Size | Form | Use |
|---|---:|---|---|
| `Ornith-1.5-9B-MTP-NVFP4.gguf` | 6.5 GB | bundled | **Blackwell: fastest rung (299 tok/s, 1.38×)** |
| `Ornith-1.5-9B-MTP-Q8_0.gguf` | 9.8 GB | bundled | largest *relative* MTP gain, reference quality |
| `Ornith-1.5-9B-MTP-Q6_K.gguf` | 7.6 GB | bundled | near-lossless quant |
| `Ornith-1.5-9B-MTP-Q5_K_M.gguf` | 6.6 GB | bundled | balanced quality |
| `Ornith-1.5-9B-MTP-Q4_K_M.gguf` | 5.8 GB | bundled | superseded by `IQ4_XS` — see above |
| `Ornith-1.5-9B-MTP-IQ4_XS.gguf` | 5.45 GB | bundled (imatrix) | **best low-VRAM rung**, fits 6 GB |
| `Ornith-1.5-9B-MTP-IQ3_M.gguf` | 4.67 GB | bundled (imatrix) | 6 GB with room for context; **needs the recommended sampler** |
| `Ornith-1.5-9B-MTP-IQ2_M.gguf` | 3.87 GB | bundled (imatrix) | smallest; **⚠️ degenerates on long output** |
| `Ornith-1.5-9B-MTP-BF16.gguf` | 18.4 GB | bundled (master) | re-quantize from this |
| `mmproj-Ornith-1.5-9B-BF16.gguf` | 922 MB | vision projector | **only** for image input — costs ~1.1 GB VRAM resident |
| `mtp-head/mtp-Ornith-1.5-9B-head-Q8_0.gguf` | 2.4 GB | standalone head | attach to a base GGUF via `--model-draft` |

"Bundled" = trunk + `nextn` head in one file. The standalone head is **not a model** — loading
`mtp-head/…` directly will crash. It exists only to pair with a base Ornith-1.5-9B GGUF.

Ornith-1.5-9B is a **vision** model; pair any rung with `mmproj-…` for image input. Verified
working with MTP enabled.

## Run

```bash
# Text-only. Do NOT load the mmproj unless you are actually sending images —
# it costs ~1.1 GB of VRAM you probably need. See "VRAM" below.
# -fit off + an explicit -ngl is deliberate: MTP inverts if even one layer
# lands on the CPU, and --fit (on by default) will move layers silently.
# The sampler flags are NOT optional — see "Sampling" below. llama.cpp's
# defaults apply no repetition control and measurably increase looping.
llama-server --model Ornith-1.5-9B-MTP-Q6_K.gguf \
  --n-gpu-layers 99 -fit off --ctx-size 8192 --flash-attn on --jinja \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --presence-penalty 1.5 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

Confirm `offloaded 34/34 layers to GPU` in the startup log. If it says anything less, fix that
before judging MTP's speed — see [VRAM](#vram--mtp-requires-a-full-offload-and-the-cliff-is-one-layer-wide).

For image input, add the projector:

```bash
  --mmproj mmproj-Ornith-1.5-9B-BF16.gguf
```

`--spec-draft-n-max` is the draft depth: **2** maximizes acceptance, **3** maximizes throughput,
**4 regresses**. Same shape our 1.0 head showed, reproduced independently here.

**Standalone draft** — pair the small head with any base Ornith-1.5-9B GGUF:

```bash
llama-server --model ornith-1.5-9b-Q4_K_M.gguf \
  --model-draft mtp-head/mtp-Ornith-1.5-9B-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --n-gpu-layers 99 --flash-attn on --jinja
```

## Sampling — pass these flags, they are load-bearing

Ornith recommends **`presence_penalty 1.5`** (temp 1.0, top_k 20, top_p 0.95, min_p 0) for
general use, and `temp 0.6` / `presence_penalty 0` for precise coding. The base repo ships no
`generation_config.json`, and llama.cpp would not read one anyway — its defaults are
`temp 0.8, top_k 40, top_p 0.95, min_p 0.05, presence_penalty 0`, i.e. **no repetition control
at all**. Earlier versions of this card shipped a command line with no sampler flags, so every
user got exactly that.

Measured on `IQ2_M`, 64 samples per arm (8 prompts x 8 seeds, 1500-token generations):

| sampling | degenerate | rate |
|---|---:|---:|
| Ornith-recommended (`temp 1.0 … presence_penalty 1.5`) | 3 / 64 | **4.7%** |
| llama.cpp defaults | 18 / 64 | 28.1% |
| greedy (`temp 0`) | 24 / 64 | 37.5% |

Fisher two-sided p = 5.5e-4 vs defaults. **Low temperature makes looping worse on this family**
— the opposite of the usual advice. This is mitigation, not a cure: the rung matters more (below).

## VRAM — MTP requires a FULL offload, and the cliff is one layer wide

MTP is not free in memory: the `nextn` head adds a layer and speculation adds a draft context.
Measured resident VRAM, `Q6_K`, `--ctx-size 4096`, `--n-gpu-layers 99`, flash-attn on:

    configuration                  resident VRAM   delta
    ---------------------------    -------------   -----
    trunk only                        7.5 GB         --
    trunk + MTP                       8.2 GB       +0.7 GB
    trunk + mmproj                    8.6 GB       +1.1 GB
    trunk + mmproj + MTP              9.3 GB       +1.9 GB

`Q8_0` adds roughly 2 GB on top of every row.

> ### ⚠️ If MTP makes it *slower*, you are almost certainly not fully offloaded
>
> **MTP inverts the moment any layer lands on the CPU.** This model has **34** offloadable
> layers (32 trunk + the `nextn` head + output). Measured, `Q6_K`, ctx 8192, `-fit off`:
>
>     -ngl      no-MTP pp   no-MTP tg   +MTP pp   +MTP tg    MTP verdict
>     ------    ---------   ---------   -------   -------    -----------
>     34 / 99        8543       171.4      6537     235.1    +37%  WIN
>     33             5892        77.3      5438      56.0    -28%  LOSS
>     28             3066        20.2      2846      12.1    -40%  LOSS
>     24             2150        13.9      2044       8.2    -41%  LOSS
>
> **One layer short is enough.** At `-ngl 33` decode has already fallen 55% and MTP has flipped
> from a 37% speedup to a 28% *slowdown*. Speculation drafts and verifies every step, so each
> round trip pays the host-boundary cost again — partial offload hurts MTP roughly twice as
> much as it hurts plain decode.
>
> **The cause is `--fit`, which is ON by default.** It silently adjusts any argument you left
> *unset* — including `-ngl` — to fit device memory with a 1024 MiB margin. Enabling MTP raises
> the estimate, so `--fit` quietly shaves the layers that MTP needs. A machine reporting
> comfortable free VRAM is **not** evidence against this: `--fit` keeps usage under budget
> precisely by moving layers to the host. Confirmed in the field — this was diagnosed from a
> user report on a 16 GB card sitting at 12.6 GB used, and `-fit off` with an explicit
> `-ngl 99` fixed it.
>
> **Diagnose it:** check the `offloaded X/Y layers to GPU` line at startup. Anything but
> `34/34` explains the slowdown by itself.
>
> **Fix, in order:** pass `-fit off` together with an explicit `-ngl 99` · drop `--mmproj` if
> you are not sending images (−1.1 GB) · move to `IQ4_XS` (−1.6 GB vs `Q6_K`, and faster
> anyway) · lower `--ctx-size`.
>
> A second, rarer cause is a driver spilling VRAM into system RAM instead of failing to load
> (silent on Windows/WDDM, and possible via the GTT heap on AMD/Vulkan). Same symptom, same
> fixes — and the startup layer line still tells you which one you are looking at.
>
> **Coming from `Ornith-1.0-9B-MTP`?** The trunk files are byte-identical in size, but 1.0 has
> **no mmproj**. A 1.0 command line that fit your card can push 1.5 over — same rung, +1.1 GB —
> and with `--fit` on, "over" means layers quietly move to the CPU rather than an error.

## The finding: a grafted head was NOT good enough this time

Our [Ornith-1.0-9B head](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF) transferred
from base Qwen3.5-9B almost intact — Ornith-1.0 was a light enough fine-tune that base-Qwen's
residual stream survived, and the raw graft hit 0.74–0.76 acceptance with zero training.

**Ornith-1.5 breaks that.** Its end-to-end RL self-improvement loop moved the hidden states much
further, and the same graft lands 0.09–0.13 lower at *every* draft depth. Re-distilling the head
against Ornith-1.5's own generations (KL distribution-match, 492 steps) recovers all of it:

    n-max   graft   distilled   Δ        1.0's shipped head
    -----   -----   ---------   ------   ------------------
      2     0.636     0.767     +0.131         0.766
      3     0.528     0.663     +0.135         0.651
      4     0.473     0.583     +0.110         0.565

The lesson generalizes: **how well an MTP head transplants is a function of how far the fine-tune
moved the residual stream.** A light SFT keeps the donor head usable; a heavy RL loop does not.
Measure acceptance before assuming a graft is enough — the head loads and generates correctly
either way, so nothing but the acceptance rate tells you.

Objective matters too: on 1.0, hard-CE distillation *regressed* the graft (0.763 → 0.721) by
over-sharpening the argmax. MTP acceptance is rejection sampling against the target, so it rewards
distribution match, not token fit. KL is the correct objective; this build uses it.

## Benchmarks

**RTX PRO 6000 Blackwell (sm120), ctx 8192, flash-attn, greedy, 6-prompt code+general mix,
`-n 200`, quiet GPU. Single-stream (C=1) — see the caveat below.**

### Q8_0, n-max sweep

| config | decode tok/s | acceptance | speedup |
|---|---:|---:|---:|
| base (no MTP) | 149.6 | — | 1.00× |
| MTP n-max 2 | 252.6 | **0.767** | 1.69× |
| **MTP n-max 3** | **264.5** | 0.663 | **1.77×** |
| MTP n-max 4 | 256.2 | 0.583 | 1.71× |

### Across the full ladder, n-max 3

Every row below was measured in **one session on one box with one prompt mix**, so the rows are
comparable to each other:

| rung | size | base tok/s | +MTP tok/s | speedup | acceptance |
|---|---:|---:|---:|---:|---:|
| **NVFP4** | 6.53 GB | 216.1 | **299.1** | 1.38× | 0.599 |
| IQ2_M | 3.87 GB | 260.7 | 276.8 | 1.06× | 0.558 |
| IQ4_XS | 5.45 GB | 228.4 | 276.9 | 1.21× | 0.525 |
| IQ3_M | 4.67 GB | 233.8 | 257.0 | 1.10× | 0.507 |
| Q6_K | 7.56 GB | 171.6 | 249.7 | 1.46× | 0.541 |
| Q8_0 | 9.79 GB | 150.5 | 236.4 | 1.57× | 0.543 |
| Q4_K_M | 5.78 GB | 203.4 | 215.7 | 1.06× | 0.544 |

**NVFP4 is the fastest rung outright**, and it is not simply "4-bit is small": IQ4_XS and IQ2_M
are *smaller* and still slower with MTP on. FP4 sits on Blackwell's tensor-core GEMM path, where
MTP's parallel verify is nearly free, while K-quants and i-quants pay that verify on the dequant
path. The speedup *ratio* still grows with precision (Q8_0 1.57×) because Q8_0's baseline is the
most bandwidth-bound — but ratio and absolute speed point at different files, and what you want
to run is the fast one.

> **Why Q8_0 reads 1.57× here and 1.77× in the sweep above:** different prompt mix, and
> acceptance moved with it (0.543 vs 0.663). MTP speedup is a function of how predictable your
> text is, so compare rows *within* a table, never across the two.

### ⚠️ Q4_K_M + MTP regresses on creative prose

Acceptance tracks predictability, and it collapses on open-ended prose (0.310 at n-max 3). On
Q4_K_M that is below the break-even point — the verify costs more than speculation saves:

    Q4_K_M, n-max 3     no-MTP    +MTP     acceptance
    code                 206.6    256.4      0.702
    math                 206.8    286.3      0.820
    structured           206.6    239.1      0.631
    creative prose       206.6    159.4      0.310   <- 23% SLOWER

On Q8_0 prose still nets positive (195.8 vs 149.6) because the baseline is slower to begin with.
**If your workload is mostly long-form creative writing on Q4_K_M, run without `--spec-type`.**

### Prompt processing

The tables above are **decode**. Prompt processing was not measured for the initial release;
it is measured here because MTP's cost lands there. `Q6_K`, ctx 16384, fully GPU-resident:

| prompt tokens | no-MTP pp tok/s | +MTP pp tok/s | no-MTP tg tok/s | +MTP tg tok/s |
|---:|---:|---:|---:|---:|
| 2,548 | 8,059 | 6,793 | 171.8 | 231.9 |
| 10,951 | 7,982 | 7,227 | 168.3 | 272.0 |

**MTP costs 10–28% of prompt-processing throughput** and pays it back in decode. That is the
whole of its inherent cost when the model fits in VRAM. A prompt-processing drop measured in
*multiples* is the memory-overflow case above, not this.

### Methodology caveat — read before quoting these

These are **single-stream (C=1)** numbers, which our house rule normally bars from a model card,
because speculative-decoding wins are known to compress or invert under concurrent load (our
[dFlash finding](https://protolabs.studio/lab): a +43% single-stream win became 3× *slower* than
MTP at C=32). We publish C=1 here because llama.cpp/GGUF deployment is overwhelmingly single-user
and local, which makes C=1 the honest representative regime for this artifact — **but do not carry
these numbers over to a batched server.** No concurrency sweep was run for this release.

## Scorecard

Measured on the **NVFP4 build of these same weights**
([`protoLabsAI/Ornith-1.5-9B-NVFP4`](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-NVFP4)) —
the quantization differs from the GGUF rungs here, so treat these as characterising the
*model*, not any specific rung. Judge-free except claw (independent cloud judge, 0 fallbacks):

    axis            score   kind                detail
    --------------  -----   ------------------  ------
    function_call   0.963   schema-checked      52/54 · best on our internal board
    claw            0.675   agentic/LLM-judged  10 tasks · robustness 1.00 · safety-clean
    reasoning_hard  0.611   solver-verified     5/9 full-pass
    livecodebench   0.115   exec-graded         hard-only, 30 problems, thinking-off

**Tool calling is this model's strength** — 0.963 is the highest function_call score across
~26 scorecards we have run, spanning 9B to 397B including dedicated coder models.

**Code generation is its weakness, and it is the model rather than any quantization.**
12 of 30 problems earned partial credit, none passed every test, and **13 of 30 consumed the
entire 32,768-token budget** deliberating without emitting working code. Users report the same
independently across the Ornith-1.5 family. Enabling thinking does not rescue it: paired on
identical problems (on the 35B sibling) thinking-on scored *worse* than thinking-off, 0.129 vs
0.329, while exhausting the budget on 6 of 7.

If you are picking a rung for agentic/tool work, any of these are a good choice. If you are
picking one for code generation, this family is not the right model at any precision.

## "Lossless" — read this

MTP speculative decoding is **distribution-lossless**: every drafted token is verified against the
target, so the output distribution is unchanged. It is **not bitwise-identical** to plain decode at
greedy/temp 0 — the batched verify computes target logits in a different floating-point reduction
order than sequential decode, which can flip a greedy argmax and fork the text. Both outputs are
equally valid; this is expected llama.cpp behavior, not a defect of these weights.

## Troubleshooting: `wrong number of tensors expected 442 got 427`

The gap is the 15 `mtp.*` head tensors. This happens if you convert the **base**
`ornith-ai/Ornith-1.5-9B` directly without grafting a head first: the base keeps
`mtp_num_hidden_layers: 1` in `config.json` but ships none of the `mtp.*` weights, so the converter
declares a `blk.32` MTP layer and leaves those 15 tensors empty.

**Fix:** graft the head into the trunk before converting, then convert with no `--mtp` flag. (Only
4 of the 15 land as `blk.32.nextn.*`; the other 11 become ordinary `blk.32.*`, so `grep nextn`
shows 4 but the head is complete.) Or run the stock base GGUF with
`--model-draft mtp-head/mtp-Ornith-1.5-9B-head-Q8_0.gguf`.

**MTP-baked GGUFs need recent runtimes.** Old Ollama (≤~0.30) fails with `layer 32 missing
attn_qkv`; update and re-pull.

## How these were built

```bash
# 1. graft Qwen3.5-9B's 15 mtp.* tensors into the Ornith-1.5 trunk
python graft.py --donor Qwen/Qwen3.5-9B --target ornith-ai/Ornith-1.5-9B \
                --out ornith-1.5-9b-mtp-graft --dtype bfloat16
# 2. corpus = Ornith-1.5's OWN generations (3942 samples, no-think, T=0.7)
python gen_corpus.py --url <served-1.5> --model ornith15 --out corpus.jsonl
# 3. distill: freeze base, train ONLY the 15 mtp.* tensors, KL objective
python distill.py --config configs/ornith-1.5-9b.yaml     # 492 steps, loss 0.889 -> 0.357
# 4. convert (remaps mtp.* -> blk.32.nextn.* automatically) + quantize
python convert_hf_to_gguf.py ornith-1.5-9b-mtp --outfile ...-BF16.gguf --outtype bf16
llama-quantize ...-BF16.gguf ...-Q4_K_M.gguf Q4_K_M
```

Recipe: [`experiments/mtp/`](https://github.com/protoLabsAI/lab) — the scripts retarget to any
Qwen3.5-family fine-tune by swapping a config.

## The i-quant rungs, and what the MTP head does at low bit depth

The `IQ` rungs are i-quants (importance-matrix calibrated) with the **MTP head pinned to Q8_0**.
That pin is load-bearing for a non-obvious reason: a plain forward pass never activates the
`nextn` head, so the importance matrix contains **no data at all** for those 15 tensors. Left
unpinned they would be i-quantized blind — on the one tensor group where that costs the most,
since a degraded draft loses acceptance on every token. (`output.weight` and `token_embd.weight`
are likewise absent from the imatrix and fall back to llama.cpp's defaults.)

Calibration corpus: ~2 MB, 70% Ornith-1.5-9B's own generations (agentic/business, coding,
general chat — the same corpus the MTP head was distilled against) and 30% literary prose. The
prose share is deliberate: instruct output is register-narrow and creative writing is the first
thing to go at 3 bits.

**MTP does not invert at low bit depth.** The Q8_0 → Q4_K_M decay (1.57× → 1.06× on the ladder mix) looks like a
trend heading for a regression at 3 and 2 bits. It is not one — acceptance holds in a 0.51–0.60
band across the entire ladder and every rung is net-positive with the head on.

### Coherence at low bit depth

Needle recall + degeneration detectors at 4K / 16K / 32K context, plus a verifiable word problem:

| rung | 4K | 16K | 32K | word problem (answer 17:05) |
|---|---|---|---|---|
| NVFP4 | clean | clean | clean | correct, with a distance check |
| IQ4_XS | clean | clean | clean | correct, with a distance check |
| IQ3_M | clean | clean | clean | correct, with a distance check |
| IQ2_M | clean | clean | clean | correct, with a distance check |

Needle exact at every depth on every rung. IQ2_M solves the two-train problem correctly *and
verifies its own answer* at 2.7 bpw.

> ### ⚠️ That table is not enough, and an earlier version of this card over-claimed from it
>
> Those are **short** outputs. Degeneration on this family scales with *generation length*, not
> context depth — so a needle probe can come back perfectly clean on a rung that falls apart
> when asked for 1500 tokens of prose. This card previously called `IQ2_M` "genuinely usable"
> and "still coherent" on the strength of the table above. That was wrong.
>
> Re-measured with long generations (1500 tokens, 8 prompts x 8 seeds = 64 per cell, three
> overlapping degeneration detectors):
>
> | rung | Ornith-recommended sampling | llama.cpp defaults | Fisher p |
> |---|---:|---:|---:|
> | `IQ4_XS` | 0 / 64 | 0 / 64 | 1.0 |
> | `IQ3_M` | 0 / 64 | **5 / 64** (7.8%) | 0.058 |
> | `IQ2_M` | **3 / 64** (4.7%) | **18 / 64** (28%) | 5.5e-4 |
>
> A wider sweep — every sampling arm, context depths to 31k, thinking on and off, 316
> generations in all — splits the same way:
>
> - every rung from `IQ4_XS` up: **0 failures out of 234**
> - `IQ2_M`: **14 out of 82**
>
> Fisher p = 2.6e-9. `Q8_0`, `Q6_K`, `Q4_K_M` and `IQ4_XS` never degenerated once.
>
> What IQ2_M emits when it goes: `summer summer summer …` x114 · `__class__` x66 ·
> `from typing import List, Optional, TypeVar, Generic, Iterator` x12. Not a marginal metric
> call.
>
> **Guidance:** `IQ4_XS` is safe at any sampler. `IQ3_M` is fine *with the recommended sampler*
> and marginal without it. `IQ2_M` should be treated as a size-constrained curiosity, not a
> general-purpose rung — the recommended sampler cuts its failure rate 6x but does not clear it.
>
> Harness, raw JSONL and the full write-up:
> [`experiments/quantize/looping/`](https://github.com/protoLabsAI/lab).

### A different failure that also gets called "looping"

On hard coding problems this model will argue with itself — *"Hmm. Wait no. Let me
reconsider… Actually, the—"* — until it consumes the entire token budget. We reproduced one at
**32,768 tokens with thinking off**, and its repetition score was near zero: it never repeats,
it just never stops.

We tried to fix this with sampling and **it does not work.** A graded LiveCodeBench re-run on
these weights — 30 hard problems, 3 trials per arm — comparing our eval's temp 0.2 against
Ornith's documented temp 0.6 for precise coding:

| sampling | LCB mean (3 trials) | hit the 32k cap | mean tokens |
|---|---:|---:|---:|
| temp 0.2 | 0.128 / 0.155 / 0.147 → **0.143** | 12.3 / 30 | 14,400 |
| temp 0.6 | 0.125 / 0.168 / 0.095 → **0.129** | 14.3 / 30 | 17,100 |

Paired per-problem across all 30 tasks, **p = 0.69** — no effect, and temp 0.6 was if anything
slightly worse. An earlier version of this card recommended temp 0.6 for coding here on the
strength of an n=8 probe on the `Q6_K` GGUF; that signal **did not replicate** at proper power
and the recommendation is withdrawn.

So: sampling fixes the *repetition* failure (see the rung table above) but not this one. Every
capped problem still emitted an extractable code block (`no_code_in_budget = 0`), so the low
score is wrong answers, not missing ones. If your looping is specifically on code generation,
no sampler setting will help — this family is strong at tool calling and weak at code.

**Want the vLLM build?** [`protoLabsAI/Ornith-1.5-9B-NVFP4`](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-NVFP4)
— W4A4 NVFP4 for vLLM on Blackwell, same distilled MTP head, vision verified against the bf16
source.

## Provenance & license

- **Base:** `ornith-ai/Ornith-1.5-9B` (MIT) — a dense Qwen3.5-9B-architecture hybrid
  (linear + full attention) VL fine-tune, trained with end-to-end RL self-improvement.
- **MTP head:** grafted from `Qwen/Qwen3.5-9B` (Apache-2.0), then KL-distilled against
  Ornith-1.5-9B's own hidden states.
- These GGUFs derive from both; **MIT**. Built by [protoLabs.studio](https://protolabs.studio).
