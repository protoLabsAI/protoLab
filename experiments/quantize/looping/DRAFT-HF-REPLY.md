Reply to Ornith-1.5-9B-MTP-GGUF discussion #4 (shelterx).

---

Which rung are you on? We went and measured this properly, and it comes down to that plus your sampler.

Long generations (1500 tokens), 8 prompts x 8 seeds = 64 per cell, three degeneration detectors:

| rung | Ornith-recommended sampling | llama.cpp defaults |
|---|---:|---:|
| IQ4_XS | 0 / 64 | 0 / 64 |
| IQ3_M | 0 / 64 | **5 / 64** (7.8%) |
| IQ2_M | **3 / 64** (4.7%) | **18 / 64** (28%) |

We also ran a wider sweep — every sampling arm, context depths up to 31k, thinking on and off — 316 generations in all.

- Every rung from IQ4_XS up: **0 failures out of 234**
- IQ2_M: **14 out of 82**

Q8_0, Q6_K, Q4_K_M and IQ4_XS never degenerated once.

**If you're on IQ2_M, move to IQ4_XS.** It's smaller than Q4_K_M, faster, and clean at any sampler.

Our card said IQ2_M was "genuinely usable" and "still coherent." That was wrong, and it's now fixed. The claim came from a coherence probe that only did short needle recall — degeneration here scales with output *length*, not context depth, so the probe never generated far enough to see it.

**The sampler matters independently of the rung.** llama.cpp defaults to `presence_penalty 0` — no repetition control at all. Ornith's card recommends 1.5, but the base repo ships no `generation_config.json`, so nothing carries it through to any GGUF runtime. Our Run block didn't pass sampler flags either. That's fixed too:

```bash
llama-server --model Ornith-1.5-9B-MTP-IQ4_XS.gguf \
  --n-gpu-layers 99 -fit off --ctx-size 8192 --flash-attn on --jinja \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --presence-penalty 1.5
```

Worth knowing: **low temperature makes looping worse on this family** — greedy was our worst arm at 38%. Backwards from the usual advice.

One caveat, in case it's what you're actually hitting. There's a second failure people also call looping: on hard coding problems this model argues with itself — *"Hmm. Wait no. Let me reconsider..."* — until it burns the whole budget. It never repeats, it just never stops.

We tried to fix that one with sampling and it doesn't work. Graded LiveCodeBench re-run on these weights, 30 hard problems, 3 trials per arm:

| sampling | LCB mean | hit the 32k cap | mean tokens |
|---|---:|---:|---:|
| temp 0.2 | 0.143 | 12.3 / 30 | 14,400 |
| temp 0.6 (Ornith "precise coding") | 0.129 | 14.3 / 30 | 17,100 |

Paired per-problem, p = 0.69 — no difference, and 0.6 was slightly worse. (I'd briefly suggested temp 0.6 here off a small n=8 probe; it didn't hold up at proper power, so ignore that.)

Every capped problem still emitted a code block, so the low score is wrong answers rather than missing ones. Strong at tool calling, weak at code gen — no sampler setting changes that.

If none of this matches — are you on Ollama or LM Studio? Context shift on? Does it start only after a long multi-turn session fills the context? We couldn't reproduce upstream's "recursive past ~22k" report single-turn, so that one's still open and we'd like to.

Harness and raw data: `experiments/quantize/looping/` in github.com/protoLabsAI/lab
