# System One, locally — using Jev as the reference point

**Status: PHASE 1 COMPLETE (local arms + deferral) — see RESULTS.md. No cloud calls made. No
spend.** Exit criterion met: calibrated local decision heads, tier-0 baselines, and a
cross-domain held-out calibration test (`T-loso`). Open for a go-ahead: the paid Jev reference
arm (`jev_arm.py`, written, never run, no key on the box).

Prompted by TypeSafe AI's **Jev** (early access 2026-09-15; Sam Witteveen covered it
2026-09-18). Jev is interesting and **not adoptable here** — the experiment below is about the
*pattern*, not the vendor.

## What Jev actually is (from primary sources, not the video)

A "System One" model: unstructured text + **typed questions** in, a typed value + calibrated
probability out, in **one parallel-sampled forward pass with no chain-of-thought**. Trained
with what they call **RLCD** (Reinforcement Learning for Calibrated Decisions). It gives up
string generation entirely.

```
model         jev-1.13.0        one model, no variants
context       64k
price         $0.042 / MTok input · output FREE
throughput    250,000 tok/s · 1,200 req/min
latency       70-500 ms end-to-end (claimed)
cardinality   <=255 options per question; beyond that needs a 2-stage score-then-choose
```

**Their headline numbers need heavy discounting, by their own admission.** The "193.6x faster,
444.6x cheaper" figures come from workflow evals that TypeSafe says were "made by individuals
on our model capabilities team, so some bias could exist", are "on the higher end of real world
gains", and — critically — the **LLM baselines were run through TypeSafe's own System One
wrapper**, which they concede "tends to be slower and more expensive than giving decisions
without probabilities". The reference point is "the average of GPT-6 Astra and Fable 5.1".
No standard public benchmarks (MMLU etc.) are reported at all.

The "0% hallucination" claim is **definitional, not empirical** — they say so: "Schema matching
is guaranteed." Constrained decoding gives us the same guarantee. It says nothing about whether
the chosen option is *correct*.

## Architecture — resolved (it is NOT masked diffusion)

Flagged this as an open question because third-party write-ups compare Jev against **MDLM**.
That comparison is a *contrast*, not a description. Jev's "Parallel Sampler" computes the
probability distribution **across all defined choices in a single forward pass**; masked
diffusion reaches parallelism a different way, by initialising a sequence of `[MASK]` tokens and
iteratively denoising over multiple steps. Jev does one pass, no denoising loop.

**Consequence for us: the parked DiffusionGemma work does NOT apply, and nothing exotic is
needed.** "Distribution over a fixed option set in one forward pass" is precisely
`max_tokens=1` + `top_logprobs` over the option tokens. Our replication path gets *shorter*,
not longer.

TypeSafe discloses no further architecture detail — no layer count, parameter count, base
model, or whether it is encoder- or decoder-derived. Treat every mechanism claim beyond
"single forward pass" as unverified.

### The three primitives

```
choice    categorical selection over <=255 options     -> "billing"
score     continuous numeric                            -> 1.4
noul      boolean-with-probability                      -> 0.95
```

All three return calibrated probabilities, which is the only part we cannot trivially clone.

## Why we are not adopting it

**Closed weights, cloud-only, no roadmap.** `docs.typesafe.ai/models` has no mention of open
weights, self-hosting, on-prem or VPC, and states "the same weights serve every account". No
timeline for either has been published. Distribution is their API, Vercel AI Gateway and
OpenRouter (`typesafe/jev-1.13`). That fails our local-first default, and calibration *is* the
product here — there is no "open the small one" play, because the small one **is** the product.

### Update 2026-09-20 — the open replicas arrived first

TypeSafe still publishes no weights, but two MIT-licensed replicas did, within five days of
launch: [`GitHub30/OpenJev`](https://github.com/GitHub30/OpenJev) (wraps any HF instruct model:
candidate loglikelihoods in one forward pass, temperature scaling, optional LoRA on a proper
scoring rule; SDK-compatible with `typesafe-sdk`; **no benchmarks reported**) and
[`mithalouni/system-one-open`](https://github.com/mithalouni/system-one-open) (Gemma 3 270M full
/ Gemma 4 E2B LoRA, trained and served on Modal; reports 76.7% on a strict subset of TypeSafe's
public eval, 74.8% on held-out task types, ECE 0.003 on demo families).

This does not change the adoption call — it *validates the design of this experiment*.
OpenJev's inference path is exactly the `tiny-grammar` arm below and its calibration step is
exactly our `T-in`, so our numbers are the measurement neither repo ships. The lesson in
Finding 7 (temperature cannot reorder a binary decision) applies to every one of them.

## The actual question

Jev's contribution is not "constrained output" — we already have that (xgrammar on the vLLM
lanes; GBNF in the llama.cpp fork now built at `~/dev/llama-prism`). It is **calibration**: a
probability you can threshold on. So:

> **Can a tiny local model with grammar-constrained single-token decoding plus post-hoc
> calibration match a purpose-trained decision model on routing-type tasks — specifically on
> deferral quality, not just accuracy?**

That is the whole ballgame in production. A decision model earns its place by **knowing when to
hand off to a big model**. Accuracy alone cannot measure that; risk-coverage can.

## Design

**Arms** — all local, all one forward pass, all schema-constrained:

```
arm            how                                             cost/1M decisions
embed-head     Qwen3-Embedding-0.6B (:8001) + logistic head    ~free, already serving
tiny-grammar   Qwen3.5-2B / 4B, GBNF or xgrammar, max_tokens=1,
               decision read from top_logprobs                 ~free, local
smart-lane     :8041 Qwen3.8-27B-NVFP4, same constraint        upper bound, expensive
jev            OpenRouter typesafe/jev-1.13                    REFERENCE ONLY - see Spend
```

**Suites** — mirror the published Jev demos so the comparison is on their home turf, plus one
of ours: language ID · sentiment score · support-ticket routing · PII detect · prompt-injection
detect · tool selection · **our gateway's own alias-routing decisions** (real traffic shape,
from `models/alias_tiers.yaml`).

**Metrics** — accuracy/macro-F1 is table stakes; the real ones are calibration and deferral
(`metrics.py` for accuracy + calibration, `bootstrap.py` for the calibration CIs, `defer.py` for
everything deferral including the cascade):

```
ECE (15-bin) + reliability diagram    is the probability meaningful at all
Brier score                           proper scoring rule; ECE alone is gameable
risk-coverage curve / AURC            THE headline - accuracy vs % deferred
selective accuracy @ 90/95% coverage  what a router would actually operate at
p50/p99 end-to-end latency            measured locally, concurrency-swept
cost per 1M decisions                 local = GPU-seconds, not tokens
```

**Calibration methods** to fit on a held-out split: temperature scaling, Platt, isotonic.
This is the cheap lever their RLCD replaces with training — the experiment is whether cheap
wins.

## Exit criterion (per the lab cycle)

Model artifact (a calibrated local decision head) + tier-0 baselines + **cross-domain held-out
eval** — i.e. calibration fitted on some suites must hold on a suite it never saw. Anything
less is fitting noise.

## Rules carried in from prior work

- **Single-trial deltas are not admissible.** Report spread across trials, and required n.
- **Never token-starve a probe.** A decision arm runs at `max_tokens=1` by construction, but
  the smart-lane arm must not be effort-pinned into a termination failure.
- **Measure through the gateway** for anything that claims to represent caller behaviour.
- Speed numbers concurrency-swept; C=1 is banned from anything published.

## Spend

**Nothing runs against a paid API without an explicit go-ahead.** The Jev reference arm is
optional and the whole point is that the experiment stands without it. If taken: input-only
pricing at $0.042/MTok makes a few-thousand-decision gate a matter of cents, but it is still a
deliberate exception to local-first, and the data leaves the network.

## What would change the answer

- ~~TypeSafe publishes weights or an on-prem path~~ — **partially answered 2026-09-20**: not
  TypeSafe, but two MIT open replicas of the pattern now exist (see above). The pattern is
  reproducible locally today; the *trained* calibration is still the only unreplicated part.
- Independent third-party benchmarks appear — every number today is first-party.
- Our own result comes out *badly*: if a calibrated 2B cannot approach a purpose-trained
  decision model on AURC, that is itself the finding, and a genuinely interesting one.
