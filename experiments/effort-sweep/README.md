# reasoning_effort sweep — Qwen3.8-27B smart lane

**Question:** where is the brevity/quality sweet spot for `reasoning_effort` on the prod smart
lane, and is the lane's shipped default the right operating point?

**Why it matters:** `models/serve-qwen38-27b.sh` serves this lane with thinking ON and
`reasoning_effort=xhigh` **by default**. Every caller that does not pin an effort gets xhigh.
[[reference_qwen38_effort_ceiling]] already found xhigh can become a think-loop on hard tasks;
this measures the whole curve instead of one failure mode.

## What the lane actually accepts

Probed through the gateway — the lane rejects everything else with a 400:

```
low | medium | xhigh (default)          + `none` (thinking off)
```

`minimal`, `high`, `max` are **not** valid on this lane. (`max` is a DSV4/jasl-lane knob; it
does not carry over.) Anything written against those values is a silent no-op at best.

## Method

- **Through the gateway** (`protolabs/smart` on `ava:4000`), per
  [[feedback_measure_through_the_gateway]] — which also load-balances the `:8041`/`:8042`
  replica pair, so the sweep uses both cards.
- **Judge-free quality on both suites** — nothing here is scored by an LLM:
  - `lcb_medium` — LiveCodeBench `release_v6`, difficulty `medium`, contest_date ≥ 2025-01-01,
    deterministic first-12 by (date, id). Execution-graded, partial credit = fraction of the
    problem's tests passed. **Medium, not hard, is the quality axis** — see below.
  - `reasoning_hard` — the 9 solver-verified tasks in `evals/tasks/reasoning_hard/`, scored
    with the repo's canonical `graders.match.MatchGrader` (so the numbers stay comparable to
    the board, and the LaTeX/markdown normalisation is applied — literal matching has produced
    false zeros here before).
- **3 trials per item per effort** — [[feedback_lcb_single_trial_noise]]: LCB moves 8–14/30 on
  identical weights, so single-trial deltas are noise.
- **Sampling held CONSTANT across arms** (temp 0.6 / top_p 0.95 / top_k 20, the Qwen3.8
  thinking preset) so effort is the only variable. Note this means the `none` arm is not at its
  own recommended non-thinking preset — deliberate, to avoid confounding the comparison.
- **Prod token budget**: `max_tokens=32768`, per [[feedback_eval_prod_token_budget]].
- **Brevity is measured in tokens, not characters** — reasoning and answer streams are counted
  on the lane's own `/tokenize` endpoint, so the split is exact.

### Reading the brevity numbers

"Brevity" is two different things and the table separates them:

- `reason_tok` — tokens burned in the think block. Cost, invisible to the caller.
- `answer_tok` — tokens in `content`. What the caller actually reads.
- `compl_tok` — the bill (`usage.completion_tokens`).

These do not move together. At `effort=none` the model emits **zero** reasoning tokens and
writes its working into `content` instead — it is not thinking less, it is thinking *out loud
in the answer*. A drop in reasoning tokens is therefore not automatically a saving.

### Caveats

- **Latency is measured under 16-way concurrent load**, not single-stream. It is a secondary
  signal; tokens are the primary brevity metric. (Single-stream latency numbers are banned
  from cards anyway — [[feedback_speed_numbers_honest]].)
- **Tool-calling is NOT covered.** Both suites are single-turn text. The lane's agentic traffic
  (`function_call`, claw) may respond to effort differently; that is a separate sweep.
- `xhigh` is both an arm and the lane's default, so the "paired vs xhigh" block reads as
  *"what changes if a caller pins this instead of taking the default."*

## Files

```
sweep.py        generate + grade. --efforts/--trials/--lcb-limit/--workers
grade_only.py   re-grade an existing run from raw.jsonl (generation is the expensive half)
report.py       aggregate -> brevity/quality table + paired permutation test vs xhigh
run-<ts>/       manifest.json (exact config) · raw.jsonl (every generation) · results.json
```

⚠️ **`raw.jsonl` is NOT committed** — the repo ignores `**/*.jsonl` (and `*.log`), and these
runs are ~9 MB of model output. `results.json` **is** committed and carries every field the
analysis uses (per-run score, reasoning/answer/completion tokens, latency, finish_reason), so
every number in this README reproduces from what is in git. What does not survive is the model's
full generated text, which is what `grade_only.py` needs — so **re-grading only works on the
box that produced the run**, at the paths above. Regenerate rather than assume it is there.

Statistics: deltas are **paired per item** (trials averaged first) and tested with an
exact sign-flip permutation test — no normality assumption, and small-n is reported honestly
per [[feedback_underpowered_is_not_null]].

## Side finding: LCB *hard* is a termination test, not a quality test

`hard` was the first choice for the quality axis and had to be **dropped from the main sweep**.
At the prod 32k budget the thinking arms mostly never produce an answer, so the suite scores
"did it terminate" rather than "was it right" — a floor, not a measurement.

Item-matched (only the 3 problems where all four arms ran; 6 runs per arm per problem,
pooled across two runs):

```
effort    n   finished   hit 32k cap   never returned
none     18      18            0              0
low      18      17            1              0
medium   18      17            1              0
xhigh    18       6            0             12
```

`xhigh` failed to return on **12 of 18** hard-problem runs — not capped at 32k, simply still
generating when the client gave up (1800 s in run 1, **3600 s** in run 2; raising the timeout
did not rescue them). `none` finished 18/18, spending ~7.4-7.9k tokens and writing its working
into the answer.

⚠️ **Honest limits on this table.** Only 3 distinct problems are item-matched, because both
source runs were stopped early — this is a strong effect on a small item set, not a rate you
should quote to 2 significant figures. The per-arm *token* averages from those runs are also
biased and are deliberately not reproduced here: the arms that never returned contribute no
token count, so the most expensive samples are missing from exactly the most expensive arm.
Source data kept in `superseded-run-*-timeout-bias/` and
`partial-run-*-lcbhard-xhigh-nonterminating/`.

**Reusable lesson:** a client timeout is not a neutral observer. Set below the worst-case
generation time (32k tokens ≈ 27 min at prod TPOT, longer under load) it silently deletes an
arm's most expensive samples and makes that arm look *cheaper* than it is. Always check whether
errors correlate with a treatment before reading any token or latency average.

## Results

`run-20260908-061817/` — 21 items x 4 efforts x 3 trials = **252 generations, 0 errors,
0 dropped samples**. Judge-free throughout.

```
reasoning_hard (9 items x3)   score  reason_tok  answer_tok  compl_tok  lat p50  capped
none                          0.796           0        4128       4129    50.7s      0%
low                           0.926        3637         653       4292    44.9s      0%
medium                        0.926        2630         705       3337    47.7s      0%
xhigh  (lane default)         0.759       12884         476      13362   175.6s     19%

lcb_medium (12 items x3)      score  reason_tok  answer_tok  compl_tok  lat p50  capped
none                          0.792           0        4073       4074    47.0s      0%
low                           0.983        5782         519       6302   109.4s      0%
medium                        0.989        6613         538       7154   121.3s      0%
xhigh  (lane default)         0.972       14028         288      14318   277.7s      6%
```

### The finding: xhigh does not reason worse — it fails to terminate

Conditioning the score on whether the run finished separates the two explanations completely:

```
                    score (all)   score | finished   runs capped   score | capped
reasoning_hard
  medium                  0.926              0.926             0             —
  xhigh                   0.759              0.932             5          0.000
lcb_medium
  medium                  0.989              0.989             0             —
  xhigh                   0.972              1.000             2          0.500
```

**When `xhigh` finishes it is the best arm on both suites** (0.932 / 1.000). Its entire deficit
is the 19% / 6% of runs that burn the whole 32k budget and score 0. This is a *reliability*
failure, not a reasoning failure — and it is the same mechanism that made LCB `hard`
unmeasurable above, just at a survivable rate.

### Sweet spot: `medium`

Ties or beats every arm on both suites while costing **53–80% fewer reasoning tokens than the
lane default**, with zero cap-outs and 2.3–3.7x lower p50 latency.

`low` is statistically indistinguishable from `medium` (identical on reasoning_hard; 0.983 vs
0.989 on lcb_medium). Either is defensible; `medium` is the safer pick on code.

**Do not run `none` for code.** It is the only arm with a quality loss that clears
significance: −0.181 vs xhigh on lcb_medium (p=0.047). Cheap, and worse.

### Power — what is and is not established

Paired per-item diffs vs `xhigh`, with the item count each contrast would need for 80% power
([[feedback_underpowered_is_not_null]]):

```
contrast                        mean d    sd    dz    n have   n needed
reasoning_hard  low vs xhigh    +0.167  0.236  0.71        9         16
reasoning_hard  med vs xhigh    +0.167  0.236  0.71        9         16
lcb_medium      none vs xhigh   -0.181  0.271 -0.67       12         18
lcb_medium      med vs xhigh    +0.017  0.107  0.16       12        323
```

- The **+0.167 quality edge of low/medium over xhigh on reasoning_hard is NOT established**
  (p=0.125, needs ~16 items, we have 9). It is suggestive and consistent with the cap-out
  mechanism, but do not quote it as a result.
- `none`-vs-`xhigh` on code reaches p=0.047 at n=12 against ~18 needed — treat as real but
  borderline, not settled.
- `medium` vs `xhigh` on code needs ~323 items to resolve: that is the signature of **no
  difference**, which is the point. The case for `medium` rests on equal quality at half the
  cost, not on beating xhigh.

### Recommendation

Pin `reasoning_effort: medium`. The lane ships `xhigh` as its default, so **every caller that
does not pin an effort is paying ~2-4x the tokens for no measured quality gain and a 6-19%
chance of a run that never returns.** Changing the default is a gateway-side edit on ava.

**Not covered:** tool-calling / agentic traffic (both suites are single-turn text), and any
effect on vision. Latency is under 16-way concurrent load and is secondary to the token counts.
