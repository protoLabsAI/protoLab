---
license: other
license_name: see-licensing-section
task_categories:
  - text-classification
language:
  - en
  - multilingual
tags:
  - calibration
  - selective-prediction
  - deferral
  - risk-coverage
  - routing
  - system-one
  - decision-models
pretty_name: Deferral Bench v1
size_categories:
  - 10K<n<100K
---

# deferral-bench-v1

**88,000 probability vectors from eight decision arms — a frontier 27B among them — over the
same 11,000 items, so you can evaluate a calibration or deferral method without running a
model.**

22 tasks in 6 families (safety, sentiment, topic, nli, ordinal, language), option sets from 2 to
20 wide, single-token constrained decisions. Every arm saw identical items in identical order, so
cascades and verifier setups between arms are **simulated exactly** rather than estimated.

## Why this exists, and what it cost us to learn

The open "System One" wrappers that appeared after TypeSafe's Jev launch (2026-09-15) all ship
the same recipe — score candidate options' loglikelihoods in one forward pass, softmax, optionally
fit a temperature — and none of them report benchmark numbers. The expensive part of checking
whether that temperature does anything is not the method. It is having many models' probability
vectors over identical items, with a frontier model among them.

**The first version of this benchmark had four tasks, and it produced a headline that was
wrong.** On those four, a 4B with a 20% deferral budget was statistically indistinguishable from
the 27B (−0.007 [−0.021, +0.006]). On these 22 it is clearly worse (−0.024 [−0.032, −0.016]), and
so is every other deferral rate including 50%. The per-item bootstrap was tight and honest both
times; the **task sample** was the problem. The four happened to include language ID, where a 4B
scores 0.953 against the 27B's 0.993.

That is the argument for this dataset, and the reason it is 22 tasks and not 4.

## Quickstart

```bash
pip install pandas pyarrow numpy
python scorer.py                                          # every arm, every task
python scorer.py --calibrate                              # + in-domain temperature scaling
python scorer.py --cascade Qwen3.5-4B-name --fallback smart-name
```

To score **your** method, write `probs/<your-arm>/<task>.test.parquet` with columns
`{id, y, probs}` — `probs` a list of floats in the option order from `meta/<task>.json`, `id`
matching the manifest — then `python scorer.py --arms <your-arm>`. If your method is post-hoc you
need no inference at all: load the shipped vectors, transform them, score.

## What is in it

```
manifest/<task>.<split>.parquet    id · label · label_idx · text_sha256 · text_len
meta/<task>.json                   options · question · family · split seed · source
probs/<arm>/<task>.<split>.parquet id · y · probs[] · label_mass · missing
baselines/embed-head/<task>.npz    W, b — the supervised baseline head (0.6B embedder)
reference/                         full scored outputs of the source experiment
arms.json                          what each arm is, family map, which arm saw labels
scorer.py                          ECE · Brier · AURC · sel@C · cascade · bootstrap CIs
rebuild.py                         reconstructs the texts from upstream, verifies every hash
```

`test` (300/task) is scored. `calib` (200/task) exists to fit post-hoc calibration and is never
scored. `label_mass` is the raw next-token probability mass that landed on *any* valid option
before renormalising — a free "is this arm even in-schema" signal.

**The texts are not here.** Twenty-two tasks from twenty upstream datasets carry twenty licenses;
`rebuild.py` pulls them from upstream and checks each item against its SHA-256 (verified
11,000/11,000 at build time), so a drifted dataset fails loudly instead of quietly changing what
the benchmark measures.

## Families

| family | tasks | k range |
|---|---|---|
| safety | injection, offensive, hate, spam | 2 |
| sentiment | sentiment, tweetsent, finnews, emotion | 2–6 |
| topic | routing, agnews, yahoo, dbpedia, massive | 4–18 |
| nli | rte, mrpc, qnli, snli, cola | 2–3 |
| ordinal | yelpstars, sst5, appreviews | 5 |
| language | langid | 20 |

Families matter because the honest held-out test is **leave-one-family-out**. Holding out one
task whose four siblings share its family leaks the family.

## The arms

| arm | model | readout | sees labels |
|---|---|---|---|
| `smart-name` / `smart-letter` | Qwen3.8-27B-NVFP4 + MTP | option text / `A)` | no |
| `Qwen3.5-4B-name` / `-letter` | Qwen3.5-4B bf16 (CPU) | option text / `A)` | no |
| `Qwen3.5-2B-name` / `-letter` | Qwen3.5-2B bf16 (CPU) | option text / `A)` | no |
| `Qwen3.5-0.8B-name` | Qwen3.5-0.8B bf16 (CPU) | option text | no |
| `embed-head` | Qwen3-Embedding-0.6B + softmax head | — | **yes** |

Readout: `max_tokens=1`, thinking off, next-token distribution restricted to each option's first
token and renormalised. First-token uniqueness is **enforced at build time** under every
tokenizer used, so under a grammar admitting only the option strings, P(first token) **is**
P(option) — constrained decoding computed in closed form, not approximated.

## Headline numbers

Macro-averaged by family, raw probabilities (AURC = mean error over all coverages, lower better):

```
arm                 language   nli   topic  safety  sentiment  ordinal   all 22
smart 27B     acc      0.993  0.861  0.815   0.778      0.726    0.602    0.782
              AURC     0.000  0.051  0.075   0.108      0.155    0.270    0.113
Qwen3.5-4B    acc      0.953  0.799  0.748   0.736      0.676    0.519    0.722
              AURC     0.011  0.101  0.120   0.153      0.208    0.345    0.163
embed-head    acc      0.720  0.599  0.812   0.813      0.694    0.519    0.698
              AURC     0.087  0.348  0.070   0.092      0.173    0.379    0.199
```

**The cascade**, 4B answering its confident share and deferring the rest to the 27B, paired
bootstrap within task:

```
defer%   pooled acc   delta vs 27B alone        verdict
   0%        0.722    -0.059 [-0.069,-0.050]    worse
  20%        0.758    -0.024 [-0.032,-0.016]    worse
  30%        0.768    -0.014 [-0.020,-0.007]    worse
  50%        0.778    -0.004 [-0.008,-0.000]    worse
 100%        0.782    (the 27B answering everything)
```

Deferring 20% closes **60% of the gap** for 20% of the 27B calls. A real cost lever — not parity.

## Findings this data supports

All reproducible from `scorer.py` and `reference/`:

1. **Temperature scaling cannot change a binary decision's deferral order.** For two options,
   `max softmax = σ(|z₁−z₂|/T)` is monotone in `|z₁−z₂|` for every `T>0`, so AURC and
   selective-accuracy-at-fixed-coverage are invariant. Every 2-option cell measures exactly
   `0.0000`. This matters for Jev's `noul` primitive and every replica whose calibration story is
   "we fit a temperature".
2. **ECE alone is a trap.** `Qwen3.5-2B-letter` reaches ECE 0.006 on langid after temperature
   scaling, at 10% accuracy, by flattening toward uniform — and its selective accuracy at 50%
   coverage is 0.080, *below* its overall accuracy. Its confidence is anti-correlated with
   correctness.
3. **The post-hoc scale budget is small on any arm worth deploying.** Isotonic fitted on the test
   set itself — an oracle, not a method — wins 11–27% of top-label Brier on the 27B and the 4B.
   The rest is ranking, which no post-hoc method reaches.
4. **Calibration transfer is weak but real, and needs many families to see.** A temperature fitted
   on 3 tasks and applied to a 4th was 6 better / 5 worse / 9 n.s. Fitted on 21 tasks across 5
   families and applied to a held-out 6th: **26 better / 14 worse / 8 n.s.**
5. **The ordinal family is the hardest and has the largest scale budget.** Star ratings put the
   27B at 0.602 with AURC 0.270. This is Jev's `score` primitive, and it has no public numbers.
6. **A verifier helps exactly when it is more accurate than the decider.** Ranking the 4B's
   answers by `P_verifier(the option the 4B chose)`: the 0.6B embedder wins on the three families
   where it out-scores the 4B, loses significantly on nli where it is much worse, ties elsewhere.
7. **Answer format costs more than model size.** The same 2B scores 0.503 on langid emitting
   `Polish` and 0.100 emitting `B)`.

## The reporting rule

**Never report ECE without AURC beside it.** Finding 2 is what that rule exists to catch: an arm
can be driven to near-perfect calibration while becoming worse than useless as a router. If you
propose a calibration method here, report accuracy, ECE *and* AURC — and if your task is binary,
do not claim the calibration step improved routing, because it provably cannot.

## Licensing

- **Everything this repo contains** — probability vectors, manifests, hashes, `scorer.py`,
  `rebuild.py`, the baseline head weights — is released under **MIT** by protoLabsAI.
- **Gold labels** derive from the upstream datasets and carry whatever terms those impose.
  **Source texts are not redistributed here at all.**
- Upstream sources are named per task in `meta/<task>.json` and reconstructed by `rebuild.py`;
  check each directly before using this where upstream terms matter.

The metadata `license: other` reflects that mix; it is not a claim over the upstream data.

## Known limits

- 300 test / 200 calib items per task, single seed. Bootstrap intervals cover test sampling noise
  only — calibration temperatures are fitted once on `calib` and held fixed, which makes them
  slightly optimistic.
- 22 tasks, 6 families — but `language` has only one task, so leave-one-family-out for that
  family is leave-one-task-out.
- Cardinality is capped by the readout: every task's options must have distinct first tokens, so
  there is no 77- or 150-way task here. High-cardinality decisions need trie-structured
  constrained decoding and are out of scope for v1.
- The `smart` arms read `top_logprobs` capped at 20; `missing` is shipped per item so you can
  check where that bound bit.
- CPU arm latencies are not a speed claim and none are published here.
- Cascade results use the 27B's *actual* answers on the deferred items — honest for this fallback,
  not transferable to a different one.

## Source

Built by [protoLabsAI](https://huggingface.co/protoLabsAI) from the `experiments/jev` study in the
`protoLabsAI/lab` monorepo — a local replication of the "System One" decision-model pattern.
Regenerate with `prep_data.py` + `prep_data_v2.py`, `run_all.sh` + `run_all_v2.sh`, then
`metrics.py` / `defer.py` / `calib_transfer.py` / `analyze_v1.py`.
