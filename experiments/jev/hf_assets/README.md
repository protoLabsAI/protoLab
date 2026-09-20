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
pretty_name: Deferral Bench v0
size_categories:
  - n<1K
---

# deferral-bench-v0

**Probability vectors from seven decision arms — including a frontier 27B — over the same 2,000
items, so you can evaluate a calibration or deferral method without running a model.**

Single-token constrained decisions over four option sets (2, 2, 11 and 20 options). Every arm saw
identical items in identical order, so cascades between arms are *simulated exactly* rather than
estimated.

**Scope, first, because it decides whether this is useful to you:** 300 test items per suite,
four suites, one seed. This is a **reference set for method development**, not a leaderboard.
Four suites is four leave-one-out folds — enough to establish the direction of an effect, not
enough to support a cross-task transfer claim. v1.0 widens the task set; the probability vectors
shipped here do not change when it does.

## Why this exists

The open "System One" wrappers that appeared after TypeSafe's Jev launch (2026-09-15) all ship the
same recipe — score candidate options' loglikelihoods in one forward pass, softmax, optionally fit
a temperature — and none of them report benchmark numbers. The expensive part of checking whether
that temperature step does anything is not the method. It is having many models' probability
vectors over identical items, with a frontier model among them.

That is what this repo is.

## Quickstart

```bash
pip install pandas pyarrow numpy
python scorer.py                                          # every arm, every suite
python scorer.py --calibrate                              # + in-domain temperature scaling
python scorer.py --cascade Qwen3.5-4B-name --fallback smart-name
```

To score **your** method, write `probs/<your-arm>/<suite>.test.parquet` with columns
`{id, y, probs}` — `probs` a list of floats in the option order from `meta/<suite>.json`, `id`
matching the manifest — then `python scorer.py --arms <your-arm>`. No inference needed if your
method is post-hoc: load the shipped vectors, transform them, score.

## What is in it

```
manifest/<suite>.<split>.parquet   id · label · label_idx · text_sha256 · text_len
meta/<suite>.json                  option set · question · split seed · source dataset
probs/<arm>/<suite>.<split>.parquet  id · y · probs[] · label_mass · missing
baselines/embed-head/<suite>.npz   W, b — the supervised baseline head (0.6B embedder)
reference/                         full scored outputs of the source experiment
arms.json                          what each arm is, and which one saw labels
scorer.py                          ECE · Brier · AURC · sel@C · cascade · bootstrap CIs
rebuild.py                         reconstructs the texts from upstream, verifies every hash
```

`test` is scored. `calib` exists to fit post-hoc calibration and is never scored. `label_mass` is
the raw next-token probability mass that landed on *any* valid option before renormalising — a
free "is this arm even in-schema" signal.

**The texts are not here.** Four sources, four licenses; `rebuild.py` pulls them from upstream and
checks each item against its SHA-256, so a drifted dataset fails loudly instead of quietly
changing what the benchmark measures.

## The arms

| arm | model | readout | sees labels |
|---|---|---|---|
| `smart-name` / `smart-letter` | Qwen3.8-27B-NVFP4 + MTP | option text / `A)` | no |
| `Qwen3.5-4B-name` / `-letter` | Qwen3.5-4B bf16 (CPU) | option text / `A)` | no |
| `Qwen3.5-2B-name` / `-letter` | Qwen3.5-2B bf16 (CPU) | option text / `A)` | no |
| `Qwen3.5-0.8B-name` | Qwen3.5-0.8B bf16 (CPU) | option text | no |
| `embed-head` | Qwen3-Embedding-0.6B + softmax head | — | **yes** |

Readout: `max_tokens=1`, thinking off, next-token distribution restricted to each option's first
token and renormalised. First tokens are unique within every suite, so under a grammar admitting
only the option strings, P(first token) **is** P(option) — this is constrained decoding computed
in closed form, not an approximation of it.

## Headline numbers from the source experiment

Accuracy, and AURC (mean error over all coverages, lower is better):

```
arm                suite       acc   sel@90  sel@50    AURC   majority
smart 27B          injection  0.893   0.933   0.987   0.020      0.600
smart 27B          sentiment  0.957   0.978   0.980   0.016      0.550
smart 27B          routing    0.803   0.830   0.887   0.107      0.240
smart 27B          langid     0.993   1.000   1.000   0.000      0.070
Qwen3.5-4B         langid     0.953   0.970   0.993   0.011      0.070
Qwen3.5-2B-letter  langid     0.100   0.081   0.080   0.921      0.070
embed-head 0.6B    routing    0.960   0.993   1.000   0.002      0.240
```

**The cascade.** Pooled over the four suites, a 4B answering its confident share and deferring the
rest to the 27B, paired bootstrap resampled within suite:

```
defer%   pooled acc   delta vs 27B alone        verdict
   0%        0.866    -0.046 [-0.065,-0.027]    worse
  20%        0.904    -0.007 [-0.021,+0.006]    indistinguishable
  30%        0.917    +0.004 [-0.007,+0.015]    indistinguishable
 100%        0.912    (the 27B answering everything)
```

Four findings this data supports, all reproducible from `scorer.py` and `reference/`:

1. **Temperature scaling cannot change a binary decision's deferral order.** For two options,
   `max softmax = σ(|z₁−z₂|/T)` is monotone in `|z₁−z₂|` for every `T>0`, so AURC and
   selective-accuracy-at-fixed-coverage are invariant. Every 2-option cell measures exactly
   `0.0000`. Where `T` *can* reorder (k>2) it is 5 better / 5 worse / 6 n.s.
2. **ECE alone is a trap.** `Qwen3.5-2B-letter` reaches ECE 0.006 on langid after temperature
   scaling, at 10% accuracy, by flattening toward uniform — and its selective accuracy at 50%
   coverage is 0.080, *below* its overall accuracy. Its confidence is anti-correlated with
   correctness.
3. **The scale budget is small on any arm worth deploying.** Isotonic fitted on the test set
   itself — an oracle, not a method — improves the 27B's top-label Brier by 0.010 and the 4B's by
   0.016, ~16% of their Brier. The rest is ranking, which no post-hoc method reaches.
4. **Answer format costs more than model size.** The same 2B scores 0.503 on langid emitting
   `Polish` and 0.100 emitting `B)`.

## The reporting rule

**Never report ECE without AURC beside it.** Finding 2 is what that rule exists to catch: an arm
can be driven to near-perfect calibration while becoming worse than useless as a router. If you
propose a calibration method here, report accuracy, ECE *and* AURC — and if your task is binary,
do not claim the calibration step improved routing, because it provably cannot.

## Licensing

- **Everything this repo actually contains** — probability vectors, manifests, hashes,
  `scorer.py`, `rebuild.py`, the baseline head weights — is released under **MIT** by protoLabsAI.
- **Gold labels** are derived from the four source datasets and carry whatever terms those
  datasets impose. **Source texts are not redistributed here at all.**
- Before using this in a context where upstream terms matter, check each source directly:
  `deepset/prompt-injections`, `stanfordnlp/sst2`,
  `bitext/Bitext-customer-support-llm-chatbot-training-dataset`,
  `papluca/language-identification`.

The metadata `license: other` reflects that mix; it is not a claim over the upstream data.

## Known limits

- 300 test / 200 calib items per suite, single seed. Bootstrap intervals cover test sampling noise
  only — calibration temperatures are fitted once on `calib` and held fixed, which makes them
  slightly optimistic.
- Four suites. Any leave-one-suite-out result here is an effect direction, not a transfer claim.
- The `smart` arms read `top_logprobs` capped at 20; every option landed inside the top 20 on
  every item (`missing = 0`), so the cap never bound.
- No `score` (ordinal) primitive — only categorical `choice` and boolean-shaped decisions.
- CPU arm latencies are not a speed claim and none are published here.
- Cascade results use the 27B's *actual* answers on the deferred items, which is honest for this
  fallback and does not transfer to a different one.

## Source

Built by [protoLabsAI](https://huggingface.co/protoLabsAI) from the `experiments/jev` study in the
`protoLabsAI/lab` monorepo — a local replication of the "System One" decision-model pattern.
Regenerate with `prep_data.py`, `run_all.sh`, then `metrics.py` / `defer.py` /
`calib_transfer.py`.
