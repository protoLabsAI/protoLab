# RESULTS — Daria-24B de-slop steering (2026-07-26/27)

Honest numbers, written for next-session-us. Full working notes: `DUE-DILIGENCE-2026-07-26.md`.
All evals: EQ-Bench Creative Writing v3 (32 prompts, 1 iteration), judge = `protolabs/cloud`
(DeepSeek) — **absolute rubric numbers are NOT leaderboard-comparable** (leaderboard judge is Claude);
within-table comparisons are same-judge and valid. Slop = EQ-Bench's deterministic 600-phrase index,
length-normalized per 1k words. Model: `mistral-creative-base-v0` (abliterated Mistral-Small-3.2-24B
+ voice SFT), served as Q5_K_M GGUF unless noted.

> ⚠️ **READ ROUND 3 FIRST (2026-08-18).** Every rubric number in the tables below was produced
> by the bench's own judging call, which truncates a thinking judge at 4096 tokens and silently
> drops unscored pieces (coverage 19-29 of 32, varying BY RUN). Corrected, uniformly re-judged
> numbers are in Round 3. The `zero loops` claim in item 5 below is also wrong — see Round 2.

## The final table

    config                                rubric   slop/1kw  Coherent  Meander  IncPos
    base (no steer)                        53.86      6.7      15.9      4.5      4.2
    constant CV, L24 @ scale 3             44.52      0.9      12.2      8.3      6.0
    gated clamp (L24, τ0.41, λ1.5)         47.59      4.0      14.3      5.8      8.9
    prompting baseline (dry sys-prompt)    48.58      5.1      15.0      7.5      5.4
    clamp + prompt                         45.88      4.5      14.1      8.8      3.3

n=1 run per config; bootstrap CIs on the rubric overlap across the middle four — criterion
patterns and the deterministic slop column are the signal, small rubric deltas are noise.

## What was established, in order

1. **Fine-tuning failed first** (pre-history): Gutenberg-style DPO pinned reward-acc at 1.0 —
   the human-vs-model separability pathology; it learns "human vs LLM", not craft.
2. **The v5 control vector was confounded.** Extracted from (human passage, Gemma-sentimentalized
   rewrite) pairs. Phase-0 diagnostics: it FAILED a human-text transfer test *inverted* (AUC 0.257
   ranking sentimental vs restrained human prose) and decodes to junk through the unembedding — its
   real axis is "Gemma-rewrite register vs modern-essayist register", a corpus contrast, ~8% pure
   authorship (cos with an explicitly-estimated authorship direction 0.29–0.49). The −87% slop
   result is real but is substantially *subtracting Gemma's slop vocabulary*.
3. **On-policy extraction fixes the direction** (v6): base-v0 wrote BOTH poles (sentimental vs
   austere rewrites of the same seeds, essay+fiction, judge-filtered, response-token mean-pooled).
   v6 passes everything v5 failed: authorship-cosine ≈ 0, cross-distribution transfer 1.00, human
   transfer 0.653, textbook logit-lens (fragile/vulnerable/softly ↔ said/anyway/plus).
4. **A clean direction is NOT sufficient: constant steering still collapses long-form.** At every
   effective strength, 1100-token generations degenerate into verbatim loops (TTR 0.44→0.08–0.21,
   max-4-gram to 300+). Rep-penalty does not save it. Mechanism (2026 literature + our repro):
   every-token steering writes perturbed K/V states that are re-attended forever — the perturbation
   compounds. Short vibe-tests (~400 tok) stop before the cliff, which is why it looked great.
   The ecosystem (repeng, llama.cpp CV users) largely never tests past 1000 tokens.
5. **Gated projection-clamp solves the collapse**: `h ← h − λ·relu(h·d̂ − τ)·d̂` — intervention
   vanishes once the state is at/below the boundary, so nothing compounds. τ calibrated from
   per-token projections of the contrast pairs. All 27 probe generations + all 32 benchmark pieces:
   TTR ≈ baseline, zero loops, Coherent 14.3 (constant CV: 12.2).
6. **Prompting is a strong baseline (AxBench's prediction held).** A dry-register system prompt
   matches the gated clamp on the overall rubric. The clamp's differentiated wins are flow
   discipline (Meandering 5.8 vs 7.5, Overwrought 5.2 vs 7.2) and slop (4.0 vs 5.1).
7. **Clamp + prompt composes**: prompt fixes the clamp's one artifact (mildly-warm endings reading
   incongruent against a dry body: IncPos 8.9 → 3.3, better than base) at the cost of more
   meandering and ~2 rubric points. Division of labor: clamp disciplines the body, prompt closes
   the endings.

## The defensible claims (and the ones we can't make)

CAN say: deterministic slop −33% to −87% with a user-controlled dial; a steering mechanism that
survives long-form where constant vectors collapse (novel-ish: no published tuning objective
measures repetition at depth); reversible, composable with prompting and samplers; quantified craft
cost at every operating point.
CANNOT say: beats base on a fiction rubric (nothing does — partly definitional: the rubric rewards
the emotional payoff being removed); leaderboard-comparable EQ-Bench numbers; "removes
sentimentality" as a human-general concept for the v5 vector (it's register subtraction).

## Ship decision (Josh, 2026-07-27)

Ship **(b)**: GGUF quants + the SINGLE-LAYER constant vector `creative-v5-cvec-L24.gguf` at
recommended scale 3.0 — the measured 44.52/0.9-slop operating point — with the honest card.
**Release identity = `protoLabsAI/abliterated-de-slop-test-24b`** (Josh: "it's not daria yet" —
the Daria brand is reserved for the polished successor: clean v6 direction + gated clamp once
packageable). Staged upload dir: `/mnt/data/gguf-out/abliterated-de-slop-test-24b-release/`.
⚠️ NOT the 31-layer `creative-v5-cvec.gguf` (over-steers; the original packaging bug).
The gated clamp is NOT expressible in llama.cpp's CV format → served-endpoint only for now;
**(c) llama.cpp gated-CV upstream patch scoped as its own experiment** (candidate next cycle).
Blog draft: `BLOG.md`. Nothing pushes to HF without Josh's explicit per-artifact confirmation.

## Round 2 (2026-08-17): position-dependent tau — endings FIXED, stability LOST

Question: the clamp leaves endings at the tau boundary, reading as incongruent warmth against a
dried body. The measured fix was a dry system prompt, which makes quality depend on the user's
prompt. Can tau ramping down over the tail fix endings *intrinsically* instead?

Config: body tau 0.41 / lam 1.5 (identical to `daria_gated`), tau ramping to **−1.0** between
generated tokens 770 and 1100. Run `daria_ramp_neg1__daria-ramp`, same 32 prompts, same judge.

    run                rubric  slop  ttr   IncPos  UnearnT  WellEarn  Overwr  Coher
    base                53.86   6.7  0.42     4.2     4.6     10.0      7.7   15.9
    constant CV @3      44.52   0.9  0.32     6.0     5.0      6.8      7.4   12.2
    gated clamp         47.59   4.0  0.38     8.9     9.7      7.6      5.2   14.3
    prompting           48.58   5.1  0.38     5.4     6.4      7.4      7.2   15.0
    clamp + prompt      45.88   4.5  0.33     3.3     4.4      7.0      5.4   14.1
    TAU RAMP -> -1.0    52.68   3.6  0.37     2.1     4.2      8.8      3.9   11.8

**On the rubric this is the best steered config we have measured** — 52.68 vs base 53.86, where
every other steered config costs 5–9 points. It posts the best-in-table Incongruent Ending
Positivity (2.1, beating even the prompt combo's 3.3), best Unearned Transformations (4.2), best
Overwrought (3.9), and better slop than the gated clamp (3.6 vs 4.0). The ending artifact is
genuinely fixed, with no system prompt.

**And it is not shippable, because it reintroduces the collapse the gate exists to prevent.**
Per-piece repetition (max repeated 4-gram, 32 pieces each):

    run              ttr med   min ttr   pieces max4>=10   worst 4-gram
    base                0.42      0.32        0/32               8
    gated clamp         0.39      0.13        3/32              28
    TAU RAMP -> -1.0    0.40      0.13        6/32             131
    constant CV @3      0.34      0.09        7/32             205

The medians look healthy (0.40 TTR, median max4 3.0 — indistinguishable from base), which is
exactly how this failure hides. Six of 32 pieces degenerate; the worst repeats one 4-gram **131
times**, ending in a literal `ELEN: No, you don't. / RHYS: Yes, I do.` loop for the last ~500
characters. That is the constant-CV failure mode, at roughly half its severity, in a config
whose *body* treatment is identical to the stable gated clamp.

Mechanistically this is what the theory predicts and we should have expected: driving tau to
−1.0 means the clamp fires on essentially every token in the tail, and a clamp that fires on
every token **is** constant steering. The gate's whole guarantee — intervention goes to zero
once the state is at/below the boundary — is voided by putting the boundary below the state
distribution. Round-1 probing had already hinted at it (global tau 0.20 → TTR 0.37, max4 3.3);
we read that as a mild cost and it was the same wall.

**Correction to the round-1 claim above.** "All 27 probe generations + all 32 benchmark pieces:
TTR ≈ baseline, zero loops" does not survive this measure: the recorded `daria_gated` run has
**3/32 pieces at max4 ≥ 10** (worst 28, min TTR 0.13). The gated clamp is much better than
constant steering and much worse than "zero loops". Mean/median TTR hid it — per-piece maxima
are the honest statistic, and the accept gate should be *worst-case*, not average.

### Three tail configs later: the endings artifact is solvable, the degeneration is not (yet)

Two further 32-prompt runs, same judge, same schedule (tokens 770→1100):
`daria_ramp_zero` (tau → 0.0, i.e. stop short of the pole) and `daria_lamramp4`
(tau pinned at 0.41, **lam** ramped 1.5 → 4.0).

    run                rubric  slop  ttrmed  minttr    bad  worst | IncPos UnearnT WellEarn Overwr Coher
    base                53.86   6.7    0.42    0.32   0/32      8 |   4.2    4.6     10.0     7.7   15.9
    gated clamp         47.59   4.0    0.39    0.13   3/32     28 |   8.9    9.7      7.6     5.2   14.3
    clamp + prompt      45.88   4.5    0.36    0.08   6/32    146 |   3.3    4.4      7.0     5.4   14.1
    tau ramp -> -1.0    52.68   3.6    0.40    0.13   6/32    131 |   2.1    4.2      8.8     3.9   11.8
    tau ramp ->  0.0    56.90   3.8    0.36    0.12   4/32     51 |   3.7    5.8      9.8     4.0   14.0
    lam ramp 1.5->4.0   52.45   4.6    0.37    0.13   4/32     40 |   2.6    4.2      8.9     3.7   12.9

`bad` = pieces with a 4-gram repeated ≥10×; `worst` = the largest such count. **This is the
statistic that decides shipping, and it is the one we were not computing.**

**The endings artifact is solved, several ways over.** Every tail config beats base on
Incongruent Ending Positivity (2.1–3.7 vs 4.2) and on Overwrought (3.7–4.0 vs 7.7), while
holding slop 30–45% below base. tau→0.0 additionally holds Well-earned Lightness at 9.8 (base
10.0) and Coherent at 14.0. No system prompt required. That question is closed.

**The lam-ramp hypothesis was wrong, and the reason generalizes.** The argument for ramping lam
rather than tau was that it preserves the gate: austere tokens (below tau) stay untouched, so
nothing compounds. That half held. But the clamp *overshoots* for the tokens it does hit —
`h·d → tau + (1-lam)(h·d - tau)`, so at lam=4 a sentimental token lands at `tau - 3·excess`,
driven well below the pole and off-manifold. The gate stayed selective about WHICH tokens it
touched and stopped being gentle about HOW FAR it threw them. Same degeneration (4/32, worst
40), different route.

**The generalization, from four independent probes:** *every* intervention strong enough to fix
the endings artifact degenerates 4–6 of 32 pieces, whichever knob you turn — tau lowered, tau
ramped, lam ramped, or a prompt stacked on the clamp. Clamp geometry is not where the remaining
win is. And the plain gated clamp is not clean either (3/32, worst 28); **base is the only
config at 0/32**. On worst-case-per-piece, no steered config passes.

**Next, and it is not more geometry:** attack the degeneration directly. (a) sampler-level
DRY / repetition-penalty / XTC — July's "rep-penalty doesn't save it" was measured against FULL
collapse at constant-CV strength (worst 205); against 4/32 at worst-40 it is a different and
probably tractable problem; (b) a serve-time guard that scores max-4-gram and regenerates,
crude but 100% effective for a served endpoint at ~12% extra generation. Then choose between
tau→0.0 and lam-ramp on a **×3 repeat** — all rubric numbers here are n=1 and the CIs in this
range overlap, so no config in this table can be claimed to beat base or each other.

## Round 2 engineering: Daria runs on vLLM

The clamp is nonlinear and cannot be folded into weights, so Daria ships as weights + code.
`daria_vllm/` is a `vllm.general_plugins` entry point that hooks the decoder layer inside the
worker process: **262 tok/s aggregate at C=8** vs ~31 tok/s single-stream on the HF path, with
batching and streaming, and it composes with a quantized checkpoint since it only touches the
residual stream. Three porting traps — all of which produce a plugin that logs success and does
nothing — are documented in `DARIA.md`. Output matches the HF reference path phrase-for-phrase,
including reproducing the same rare nonsense-refusal artifact.

## Round 3 (2026-08-18): the sampler is the lever — and a judge bug that distorted every table

### FIRST: a measurement defect that invalidates the raw bench scores above

`creative_writing_bench.py` judges with **`max_tokens=4096`** (`core/conversation.py:194`).
Our judge (`protolabs/cloud` = DeepSeek) is a THINKING model: it spends that budget on
reasoning tokens and returns empty content, so the piece silently goes unscored. Coverage
therefore varies by run — and the bench reports a rubric averaged over only what it parsed:

    run                pieces  judged
    base_final             32      29
    daria_gated            32      29
    daria_ramp_zero        32      23
    daria_lamramp4         32      21
    daria_tau0_rep115      32      19   <- the best-scoring run had the WORST coverage

Comparing a 19/32 run against a 29/32 run is not a comparison. **Every headline rubric number
in the sections above is affected, including the July table.** Fixed by re-judging all 224
stored pieces uniformly with a 16k budget + reasoning_content fallback (`daria_judge.py`):
**224/224 scored, zero failures.** Corrected numbers below; `was` = the bench's truncated value.
This is the same trap as [[feedback_eval_prod_token_budget]], hit twice in one day — once in our
own harness, once inherited from the bench. **Check judge COVERAGE, not just judge liveness.**

    config          rubric     was   slop   ttr     bad  worst  words
    base             55.29   53.86    6.7  0.42    0/32      8   1021
    gated            50.99   47.59    4.0  0.39    3/32     28   1221
    gated+prompt     48.68   45.88    4.5  0.36    6/32    146   1317
    tau-1.0          51.24   52.68    3.6  0.40    6/32    131   1302
    tau0             55.81   56.90    3.8  0.36    4/32     51   1263
    lam4             50.40   52.45    4.6  0.37    4/32     40   1255
    tau0+rep1.15     58.06   62.21    4.9  0.61    0/32      2    726

Corrections that matter: `gated` was UNDER-scored by 3.4; `tau-1.0` and `lam4` were OVER-scored
by ~1.5-2; `tau0+rep1.15` drops 62.21 -> 58.06. Rankings changed. Do not cite the old column.

### The sampler clears the gate where four clamp geometries could not

Screened on the DEGENERATION-PRONE subset (prompts 15/21/22/3/11/18 — the ones that fail in
3-4 of 5 recorded runs; enriched, so rates are not the 32-prompt rate), config tau0:

    arm       n  ttr med  min ttr  max4>=10  worst  words
    none     12     0.32     0.14      4/12     80   1626
    rep1.05  12     0.42     0.13      1/12     21   1164
    rep1.10  12     0.50     0.19      1/12    206   1089
    rep1.15  12     0.55     0.47      0/12      3    800
    nrng12   12     0.35     0.20      1/12     12   1163

rep1.05/1.10 are non-monotonic on worst-case (21 vs 206) — at n=12 the 0-vs-1 distinction is
weak evidence. `no_repeat_ngram_size=12` underperformed the reasoning behind it: blocking
12-grams does not stop a 4-gram recurring in varying contexts. Only **repetition_penalty 1.15**
is qualitatively different (worst 4-gram of 3 = nothing approaching a loop).

**On the full 32: tau0 + rep1.15 is the FIRST config to pass the gate — 0/32, worst 4-gram 2,
cleaner than base's own worst of 8.** It also posts the best Elegant Prose (9.5 vs base 8.6),
Incongruent Ending Positivity (1.5 vs 3.3), Unearned Transformations (2.3 vs 6.3), Amateurish
(8.4 vs 10.3), and slop 4.9 vs 6.7.

**What it costs, and this is not small:** Adherence to Instructions **11.4 -> 7.8** and Coherent
**13.8 -> 11.2**, with pieces 29% shorter (726 vs 1021 words). The adherence drop is very likely
the length: prompts request word counts and it undershoots them.

**The length confound runs the FAVOURABLE way, checked rather than assumed.** Within-run
correlation of piece length with rubric score is POSITIVE (r = +0.28 base, +0.34 here); base's
long half scores 11.41 vs its short half 10.09. Shorter is a handicap on this rubric, so
rep1.15 earned 58.06 despite a length penalty, not because of one.

**Do NOT claim it beats base.** +2.77 over base is inside the noise band we have seen on n=1
runs. The defensible claim is: *matches base on overall craft while eliminating the
degeneration that every other steered config exhibits, cutting slop 27%, and fixing the endings
artifact outright — at a real cost in instruction adherence and coherence.*

**Next:** x3 repeat of `base` and `tau0+rep1.15` to establish the noise band and confirm the
adherence/coherence costs are real; try rep 1.12-1.15 with a length nudge to recover adherence;
port the tau ramp + repetition_penalty into the vLLM plugin (vLLM supports repetition_penalty
natively as a sampling param; the tau ramp needs per-sequence output position from the forward
context under continuous batching).

## Round 4 (2026-08-18): the x3 controlled three-way — the honest decomposition

Three arms, three runs each, same 32 prompts, uniform 16k judging, EVERY knob stated
explicitly (an earlier attempt left repetition_penalty unset on the base arm and it silently
inherited the lane default of 1.15 — the "base" was not base):

    arm             runs (rubric)       mean    sd   slop   degen    words
    base            52.6, 53.2, 53.2   52.99  0.33    6.8    3/96     1109
    base+rep1.15    58.2, 58.4, 61.5   59.37  1.87    5.8    0/96      795
    tau0+rep1.15    56.8, 60.2, 60.7   59.21  2.11    4.7    0/96      757

    pooled run-to-run sd = 1.44 rubric points -> anything under ~2.9 is noise

    sampler only   base -> base+rep1.15 : +6.38 rubric | slop 6.8 -> 5.8 | degen 3/96 -> 0/96
    steering only  base+rep -> daria    : -0.16 rubric | slop 5.8 -> 4.7
    full stack     base -> daria        : +6.22 rubric | slop 6.8 -> 4.7 (-31%)

**The decomposition is unambiguous, and it is not the answer we were looking for.
`repetition_penalty 1.15` does essentially ALL of the work** — +6.38 rubric (4.4 sd above the
noise floor) and it eliminates degeneration outright. **The control vector, on top of it,
contributes zero measurable craft (-0.16, well inside noise) and a further ~19% slop
reduction.** Six weeks of clamp geometry is worth one sampler parameter plus a slop dial.

Not a length artifact: rep1.15 SHORTENS pieces (1109 -> 795 words) and within-run correlation
of length with rubric is POSITIVE (r = +0.28), so the sampler earned +6.38 against a length
handicap.

**What this retires:**
- Every rubric comparison in Rounds 1-3, and in the July table. Single runs on a metric with
  sd 1.44 and truncated judge coverage; the "wins" were noise plus a measurement defect.
- The framing that the clamp is the product. It is a **slop dial** (-19% on top of a sampler),
  and that is the entire measurable contribution.

**What survives, and it is worth keeping:**
- The de-slop effect is real, judge-free, and reproduces in every run (deterministic index).
- The long-form collapse finding and the gated clamp as the fix for its catastrophic form
  (constant CV worst 4-gram 205 -> gated 28) — still the most novel thing here.
- **A sampler parameter beat the intervention we built.** The AxBench lesson (run the cheap
  baseline first) generalizes past prompting: run the SAMPLER baseline too, before attributing
  anything to a mechanism.
- Base's own run-to-run sd is 0.33 while the steered arms sit at 1.9-2.1 — steering ADDS
  variance. Worth remembering when reading any single steered number.

**Daria's honest one-line spec:** base-quality prose (+/- noise) with ~31% less deterministic
slop and zero degeneration, at 28% shorter pieces — of which the sampler supplies the craft and
stability, and the v6 clamp supplies the last ~19% of the slop reduction.

## Round 5 (2026-08-19): the clamp keeps its job — the n=3 null was underpowered

Round 4 left the clamp with exactly one surviving claim, "a further ~19% slop reduction", and
that claim rested on **one slop number per arm**. Keep-or-drop is not decidable from a point
estimate, so the first step was to re-analyse the banked Round 4 pieces per run
(`clamp_decision.py`, no generation, no judge — the slop index is a deterministic function of
the text, so the answer was already in the files):

    arm            slop mean    sd   rubric    sd   degen   refuse
    base                6.83  0.86    10.60  0.07    2/96     0/96
    base+rep1.15        5.82  0.76    11.87  0.37    0/96     0/96
    tau0+rep1.15        4.70  0.86    11.84  0.42    0/96     0/96

    clamp contribution: -1.12 per 1k words (-19.2%), pooled sd 0.81, |delta|/sd = 1.4, ranges OVERLAP

(These reproduce the published Round 4 row exactly — 6.8 / 5.8 / 4.7 — so this is the same
quantity, just resolved per run instead of pooled.)

**At n=3 the slop delta does not clear the noise band.** It would have been easy, and wrong,
to stop there and call the clamp a placebo. `|delta|/sd = 1.4` with n=3 is a statement about
**power**, not about the effect: at d ~ 1.4, 80% power needs about **n=9 per arm**.

So we ran that. Two vLLM lanes served from the same NVFP4 weights and the same launch script
— identical `MAXLEN 32768 / REP_PENALTY 1.15 / MAXSEQS 8 / UTIL 0.22`, differing in exactly one
environment variable (`DARIA_ENABLE`) — 9 runs x 32 EQ-Bench prompts each, 576 pieces, 0
failures, 15.0 min. **No judge was used**: the open question is the slop index, and Round 4
already settled the rubric axis at -0.16.

⚠️ **The first attempt at this run was invalid, and the way it was invalid is worth keeping.**
`clamp_ab.py` built its job list arm-major (`for arm in lanes` outermost) and
`ThreadPoolExecutor` consumes FIFO, so all 288 clamp_on pieces completed before the first
clamp_off piece started — pieces 0-276 were pure clamp_on. The arms were **time-separated**,
which is precisely the confound that serving them concurrently is supposed to remove, while the
docstring and this file both claimed concurrency. Caught in review on protoLab#31, not by us.
The fix is one loop reorder (arm innermost); the numbers below are the re-run, whose arms
interleave across 572 of 576 pieces and balance 145/143 across halves.

    arm          slop mean    sd   words   degen     refuse
    clamp_off         5.79  0.52     745   0/288      0/288
    clamp_on          4.29  0.61     720   0/288      0/288

    delta        -1.50 per 1k words (-25.9%)
    permutation  p = 0.00004114 = 2/48620 — the MINIMUM attainable two-sided p at n=9/arm
    ranges       on 3.33-5.13   off 5.32-7.00   DISJOINT

**The clamp's de-slop effect is real, and fixing the defect made it cleaner rather than
weaker.** Every one of the nine clamp_on runs sits below every one of the nine clamp_off runs —
perfect separation, so the permutation test returns the smallest p it can express. Run-to-run
sd also fell (1.07/0.75 -> 0.52/0.61) because time-separation had been injecting variance that
the interleaved design removes. The effect size replicates Round 4's estimate (-25.9% vs -19%),
and it survives a properly powered, assumption-free test.

The n=3 "null" was the test being too small to see a real effect, which is the failure mode
opposite to the one Round 4 was built to catch.

Two side results from the same run:

- **The nonsense-refusal artifact is not clamp-attributable.** `refusal_probe.py`, 12 short
  prompts x 12 reps x 2 arms, arms interleaved: **0/144 in both** (95% CI 0-2.60%). (This probe
  had the same arm-major defect on its first pass; the numbers here are the interleaved re-run,
  which agreed.) DARIA.md reported it at "low
  single-digit rates on short prompts", and a 2.6% upper bound does not exclude that — but it
  does rule out the clamp as the cause at any rate above ~2.6%, and it is not common on this
  prompt set. Separating a genuine 1-2% rate from zero needs ~1000+ calls per arm.
- **Degeneration stays solved.** **0/288 in both arms** on the interleaved run.
  ⚠️ The gate is `>=10`, as this file defines it. `clamp_decision.py` originally used a strict
  `>10`, which is why an earlier draft of the Round 5 table reported the base arm as 2/96 where
  Round 4 reports 3/96 — under a sentence claiming the rows reproduced *exactly*. Fixed; they
  now do. An off-by-one in a ship gate is not cosmetic.

**DECISION: keep the clamp.** It does exactly one thing and that thing is now established.
Daria's honest spec: **the craft and stability of `base + repetition_penalty 1.15`, with ~24%
less deterministic slop.** The four serving traps in DARIA.md are the price of a measured
24% slop reduction, which is a defensible trade — where "zero measurable contribution" would
not have been.

### Held-out check: is the de-slop effect just tuning on the metric we report?

Fair challenge (protoLab#30): the tau/lam operating point was selected in
`phase2_gated_probe.py` using `slop_1kw` over the **top 600** phrases — the same index Round 5
reports. The powered p-value establishes the effect **reproduces across fresh generations**; it
does not establish that it exists on anything the clamp was not fitted to. Those are different
claims and only the first was supported.

The phrase list has **50,084** entries. Bands past 600 were never used in tuning or in any
published number, so they test the underlying claim — "the clamp reduces slop-register
language" — rather than "the clamp reduces these 600 strings". Same 576 banked pieces, no
generation, no judge (`clamp_decision.py --holdout`):

    band                   phrases     off      on    delta     pct    perm p
    tuned-on (top 600)         600    5.79    4.29    -1.50  -25.9%   0.00004   <-- tuned here
    held-out 600-1200          600    3.35    2.24    -1.11  -33.2%   0.00004
    held-out 1200-5000        3800   13.55   10.59    -2.96  -21.9%   0.00004
    held-out 5000-20000      15000   45.50   39.82    -5.67  -12.5%   0.00004

**The effect generalizes.** It holds on all three held-out bands — 19,400 phrases the operating
point never saw — at the floor p-value, the same perfect separation as the tuned band. The
declining percentage across wider bands is expected
rather than troubling: broader bands admit progressively more ordinary English that any prose
contains, which dilutes the signal without reversing it.

So the clamp is **a real register shift, not an artifact of fitting 600 strings**. That is the
strongest version of its case, and it is worth being precise about what it does NOT rescue:
**the -0.16 rubric result stands unchanged.** The clamp moves lexical register, demonstrably
and generalizably, and contributes nothing measurable to judged craft. Both halves are true at
once, and the honest spec has to carry both.

**The methodology lesson, and it is the mirror of Round 4's.** Round 4's lesson was *run the
cheap baseline first* — it caught an effect being credited to a mechanism that a sampler
parameter supplied. Round 5's is the other error: **"inside the noise band" at small n is not
a null result.** Before retiring a mechanism, compute the n its own effect size would need.
Both failures look like rigour from the inside.

## Artifacts

- Directions: `/mnt/data/abliterate/creative-vectors/` (v1–v6 + n_auth, VERSIONS.md registry)
- Pairs: `onpolicy_pairs_v6{_raw,}.json`, `auth_pairs_v1.json`, sent-score sets (same dir root)
- GGUFs: `/mnt/data/gguf-out/` (f16 + Q4/Q5/Q6/Q8/IQ3_M/IQ3_XXS + imatrix + both cvec files)
- Eval runs: `creative-writing-bench/creative_bench_runs.json` (base_final, daria_L24_s3,
  daria_gated, prompt_baseline, gated_plus_prompt)
- Scripts: `phase0_*.py, phase1_*.py, phase2_*.py, serve_gated.py, cvec_export_single.py`
