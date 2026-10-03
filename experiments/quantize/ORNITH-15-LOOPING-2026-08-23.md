# Ornith-1.5-9B looping — measured (2026-08-23)

Triggered by [`Ornith-1.5-9B-MTP-GGUF` #4](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-MTP-GGUF/discussions/4)
(shelterx): *"Ornith 1.5 9b suffers from the same looping issues that the 1.0 model has.
I haven't found a real way to actually fix it."* No rung, runtime or settings given.

**Answer: it is the rung.** `IQ2_M` degenerates; nothing above it does. 14/82 vs **0/234**,
Fisher two-sided **p = 2.6e-9**. Sampling is a second-order effect and cannot rescue 2-bit.

Harness: `llama.cpp b9870` CUDA, GPU0, `-fit off -ngl 99`, full offload confirmed, thinking off
unless stated, 1500-token generations, seed-fixed. Detector = three overlapping signals
(trailing-cycle period, duplicate 15-gram fraction, longest back-to-back repeated block), with a
self-test that must fire on a synthetic loop and stay silent on clean prose.

## The result

    rung        looped     n     rate      (sampling sweep + depth + thinking-on combined)
    Q8_0             0    70     0.0%
    Q4_K_M           0    82     0.0%
    IQ4_XS           0    82     0.0%
    IQ2_M           14    82    17.1%

Per-cell, 1500-token generations, 8 prompts per cell:

    arm                       Q8_0    Q4_K_M    IQ4_XS     IQ2_M
    llamacpp_default           0/8       0/8       0/8       2/8
    upstream_general           0/8       0/8       0/8       0/8
    upstream_coding            0/8       0/8       0/8       1/8
    greedy                     0/8       0/8       0/8       3/8
    default_plus_pp15          0/8       0/8       0/8       1/8
    upstream_nopp              0/8       0/8       0/8       2/8
    default_plus_dry           0/8       0/8       0/8       1/8

What IQ2_M actually emits — unambiguous, not a borderline metric call:

    summer summer summer summer ...                              x114
    from typing import List, Optional, TypeVar, Generic, Iterator x12
    __class__\n\n\n__class__ ...                                  x66
    it was *there*, and it was *not there*, and it was *there*    x18
    He had never once in thirty years seen the light go out.      (verbatim, to the cap)

## This corrects our own model card

The card currently says IQ2_M is *"genuinely usable"*, *"still coherent"*, with *"clean
degeneration detectors"*, and recommends it for 6 GB cards. That claim came from the release
coherence probe: needle recall at 4K/16K/32K plus a word problem — all **short** outputs. The
probe never generated long enough to reach the failure. At 1500 tokens it degenerates 17% of
the time.

**Lesson worth keeping: a coherence gate that only does short-output needle recall does not
test coherence.** Degeneration is a function of generation *length*, not context depth. Any
future i-quant ladder needs a long-generation degeneration arm before we call a rung usable.

## What is NOT the cause (for our files)

- **Chat template.** Ours embeds the full 7,756-char Jinja template. The third-party
  "Ollama-fixed" Ornith repos exist because some GGUFs fall back to raw passthrough; ours do not.
- **EOS.** Ours is `248046` = `<|im_end|>`, correct. (Upstream's `config.json` carries an
  inconsistent `text_config.eos_token_id: 248044`, which is the *pad* token — the converter took
  the right one.)
- **Sampling, at rungs >= IQ4_XS.** 0 loops in 168 samples across 7 arms *including greedy*.
- **Context depth.** 0 loops to 31,150 prompt tokens at every rung >= IQ4_XS.
- **Agentic tool loops.** 20 claw traces, 0 repeated tool calls, 0 text loops.

## Two different failures both called "looping"

Replaying the 6 LiveCodeBench problems that consumed the full 32,768-token budget on the
2026-08-22 scorecard, this time **keeping the text** (the runner does not persist generations,
which is why we never established this before):

1. **Repetition loop** — `3562`, at **Q6_K**: rep15 0.36, degenerate code emission. So genuine
   repetition is *not* exclusive to IQ2_M; hard coding problems can induce it at high precision.
2. **Self-doubt spiral** — `abc388_g` at Q6_K reproduced the 32,768-token cap exactly, but
   rep15 = 0.088 and the detector correctly says **not looped**. It is not repeating; it is
   arguing with itself:

   > *"Hmm. The correct two-pointer should ensure... Wait no. Let me reconsider... Actually, the"*

   for 32,768 tokens, with thinking **off**. No sampler fixes this. It is the RL-trained
   self-correction behaviour running away on problems past the model's ability, and it is the
   same thing our scorecard measured as LCB 0.115 with 13/30 capped.

Users experience both as "it loops". They need different answers, so the reply should separate
them.

## The sampling gap in our card (real, independent of the above)

`ornith-ai/Ornith-1.5-9B` recommends **`presence_penalty=1.5`** for general use — and ships
**no `generation_config.json` at all**. llama.cpp would not read one anyway. Its defaults are
`temp 0.8, top_k 40, top_p 0.95, min_p 0.05, presence_penalty 0.0` — no repetition control of
any kind. Our card's Run block passes **no sampler flags**, so every user of our repo gets
exactly the configuration the model author warns against. That should be fixed on the card
regardless of whether it moves the loop rate.

## Powered A/B: does the recommended sampling help at IQ2_M?

The ladder's within-IQ2_M sampling comparison was 1/16 vs 8/32 → Fisher p = 0.24, i.e.
underpowered and unquotable ([[feedback_underpowered_is_not_null]]). Re-run at 8 prompts x
8 seeds = 64 per arm:

    arm                 looped     n     rate      (IQ2_M, 8 prompts x 8 seeds)
    upstream_general       3/64          4.7%      temp 1.0 top_k 20 top_p 0.95 min_p 0 pp 1.5
    llamacpp_default      18/64         28.1%      temp 0.8 top_k 40 top_p 0.95 min_p 0.05 pp 0
    greedy                24/64         37.5%      temp 0

    upstream_general vs llamacpp_default   Fisher two-sided p = 5.5e-4
    upstream_general vs greedy             Fisher two-sided p = 6.0e-6

**Yes — 6x.** The model author's recommended sampling takes IQ2_M from 28% to 4.7%. Greedy is
the worst arm at 37.5%, so on this family low temperature makes looping *worse*, which is the
opposite of the usual intuition and worth stating on the card explicitly.

It does **not** eliminate it (3/64 remains), so this is mitigation, not a fix. The fix is the
rung. And the arm changes four terms at once (temp, top_k, min_p, presence_penalty); the
ladder's single-term cells were too small to decompose which one carries the effect, so we
should not claim "it is the presence penalty" without an isolated, powered arm.

## Untested — do not claim

- **Context shift.** The `--context-shift` arm never fired: `shift_on` and `shift_off` returned
  byte-identical output (same token counts, same rep15), because this build truncates a single
  generation at `n_ctx` rather than shifting. The hypothesis — that shift corrupts the recurrent
  state of Ornith's linear-attention layers (3 of every 4) and is what upstream's
  "past ~22k it loops forever" report describes — is **untested**, not refuted. Testing it needs
  a multi-turn harness where the *prompt* grows past `n_ctx`.
- **MTP.** Every run here was without `--spec-type`. Speculative decoding is
  distribution-lossless in principle, but its effect on loop rate was not measured.
- Ollama / LM Studio paths, and older llama.cpp builds where context shift was default-on.

## Files

    looping/loopdet.py     three-signal degeneration detector + self-test
    looping/sweep.py       7 sampling arms x 8 loop-prone prompts
    looping/depth.py       context-depth sweep on real prose filler
    looping/lcbloop.py     replays the LCB problems that capped, keeping text
    looping/ctxshift.py    long generation overrunning a small context
    looping/powered.py     64-per-arm sampling A/B at IQ2_M

## Follow-up: does harness sampling depress our LiveCodeBench score? (No.)

`run_livecodebench.py` defaults to temp 0.2 with no presence penalty, and every Ornith-1.5
scorecard ran at exactly that (confirmed in each `run.log`). Ornith documents 0.6 for precise
coding. Since low temperature is the worst regime for looping on this family, the obvious
worry was that we had measured the harness rather than the model.

**We had not.** Graded re-run, 9B NVFP4, 30 hard problems, 3 trials per arm:

    arm         trials                    mean    capped/30   avg tokens
    temp 0.2    0.128 / 0.155 / 0.147     0.143       12.3       14,400
    temp 0.6    0.125 / 0.168 / 0.095     0.129       14.3       17,100

Paired per-problem across all 30 tasks, sign-flip permutation: **p = 0.69**. No effect, and
0.6 was directionally *worse* on every axis. The published 0.115 reproduces at 0.143 across
three trials — consistent within LCB's known single-trial noise. **The score is real.**

Two things this corrects, both ours:

1. **An n=8 probe on the `Q6_K` GGUF said temp 0.6 halved the cap rate (4/8 → 2/8).** It did
   not replicate on NVFP4 via vLLM at n=30x3 — it inverted. The probe was underpowered, was
   labelled as such, and was still used to put a recommendation on a public model card before
   the powered test existed. That was the mistake: the ordering, not the probe.
2. **`no_code_in_budget = 0` on every scorecard**, so hitting the 32k cap never blocked
   grading. The mechanism we were chasing — "burns the budget, emits nothing" — does not
   exist. The model emits code every time; the code is wrong.

**The 35B never had this problem at all**: 0/30 capped, 2,818 mean tokens. Sampling cannot
explain its 0.205 either.

### And a harness failure worth recording

The first attempt at this A/B returned `mean=0.000` six times in ~10 s per run, and the runner
reported those as scores. Cause: the lane was served with `--max-model-len 32768` while LCB
requests `max_tokens=32768`, so prompt + budget exceeded context and vLLM returned **400 on
every call**. `call_model` swallows exceptions and records `0 tokens, score 0.00`.

The preflight missed it because it probed with `max_tokens=2048` — a token-cheap health check
passes while every real request fails. **Gate at the budget the eval actually uses.** Same
family as [[feedback_eval_harness_selfsabotage]] and the `llm_judge` 0.5-fallback: a config
error is being laundered into a publishable number. `call_model` should distinguish "no code
produced" from "the request failed"; it currently cannot.

## Harness changes made

`evals/runners/sampling.py` — one place that resolves and **records** sampling. Per-suite
defaults reproduce each runner's prior behaviour exactly (LCB 0.2/none, CTI 0.0/none, FC
0.0/1.5, RAG 0.7/1.5) so no board number moves; the change is that sampling is overridable
per-suite by env (`LCB_PRESENCE_PENALTY` now exists), and written into `config.sampling` in
the results JSON so it reaches `scorecard.json` instead of only `run.log`.

Wired into `run_livecodebench`, `run_ctibench`, `run_function_call`. Defaults deliberately
NOT harmonised — LCB at 0.2 with no penalty is probably wrong for thinking models, but
changing it silently would invalidate every historical LCB number without a re-baseline.
FC's local-vs-cloud sampling asymmetry (cloud APIs reject `extra_body`) is now documented in
code rather than implicit.
