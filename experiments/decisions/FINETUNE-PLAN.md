# Fine-tuning a small Qwen decision model (plan, 2026-10-05)

Goal: a Qwen3.5-{0.8,2,4}B that answers `/v1/systemone` questions in one forward pass, calibrated,
and good enough to take most decisions off the 27B. It ships through the same `s1serve.py` prompt
format, so the fine-tune is a drop-in backend. This is `experiments/jev` rung 2 (LoRA on a proper
scoring rule, evaluated on held-out task families), made concrete.

## Bars to beat (TypeSafe strict common subset, 343 pairs; reference = frontier consensus)

```
Jev (TypeSafe, closed)                    0.875
system-one-open Gemma 4 E2B (trained)     0.767   held-out tasks 0.742
our 27B one-pass, zero training           0.766   (0.738-0.793 over 8 runs)
stock Qwen 7B (jev-on-a-laptop)           0.738
Qwen3.5-{0.8,2,4}B zero-shot              to measure (step 1)
```

A 2B/4B that matches the 27B one-pass is the useful result: same decisions at a fraction of the
cost. Beating the open replica's E2B is the publishable one.

## Recipe

- **Model:** causal LM. Read the logits at the answer position over the label tokens (A–T, 0–9).
  LoRA r=32 on attention + MLP projections, bf16 base. Qwen3.5 is hybrid (Gated DeltaNet + full
  attention), so check that peft targets the DeltaNet projections correctly before a long run.
- **Prompt:** byte-identical to `s1serve.py` (system prompt, `<state>`, `<question>`, labelled
  options, thinking off). Training/serving parity is the point.
- **Loss:** cross-entropy on the gold label + KL to the 27B teacher's distribution (where we have
  one) + Brier, which is what system-one-open used for calibration. Fit a temperature on a calib
  split afterwards.
- **Option-order robustness:** shuffle option order per example. Otherwise the model learns
  positions, not options.

## Data

1. **Public decision datasets:** port system-one-open's `s1/data_real.py` loaders (MIT code; each
   dataset keeps its own license, recorded in a manifest). It has 92 tasks across qa / sentiment /
   nli / moderation / topic / judge / intent / scale / relevance / tools / agent. Cap per task.
2. **Our Jev study's 22 tasks** (`experiments/jev/data`), train splits only.
3. **Teacher soft labels:** a sample (~30k items) re-labelled by the 27B through `s1serve`. That
   gives distributions, not just argmax, which is what the open replica didn't have. Average 2
   passes per item, because the lane isn't batch-invariant (~10% flips). Run at low concurrency
   in quiet hours (11Z–17Z) so prod isn't hurt.
4. **Long workflow-style states:** the public data is short (≤2K tokens) and TypeSafe's cases run
   up to ~12K tokens of documents. That's the likely weak spot. We'd want some synthetic
   long-document decisions, kept clearly separate from TypeSafe's domains so we don't teach to
   the test.

**Held out, never trained:** TypeSafe's eval (test only), system-one-open's 23 held-out tasks, and
1–2 whole families from our 22. Report in-task vs held-out-family separately (rung 1 lesson:
calibration doesn't transfer across families).

## Ladder

```
step  what                                    compute                         est.
0     data build + manifest                    CPU                             ~1 h
1     zero-shot 0.8B/2B/4B on TypeSafe eval     CPU (HF) or a GPU window        CPU: hours for 4B
2     teacher labels (~30k items x2)            prod lanes, low concurrency     ~2-3 h quiet hours
3     0.8B LoRA smoke + full                   1 GPU                           ~1 h
4     2B LoRA                                  1 GPU                           ~3 h
5     4B LoRA (if 2B is promising)             1 GPU                           ~5 h
6     eval all + C-swept speed (vLLM serve)    1 GPU                           ~1 h
```

Token budget ~70M per run (what E2B used). Time estimates assume one RTX PRO 6000 in bf16 + LoRA,
which is roughly 0.5–0.7× an H100.

## Compute: the open question

Both cards are prod. Training next to a prod lane on the same card is ruled out (a co-tenant made
prod 2.4× slower, 2026-09-10). Options:

- **A. Quiet-window takeover of GPU0** (replica B out, `window.sh`-style guard: abort if A's
  queue builds, hard deadline, restore B on exit). 11Z–17Z fits steps 3–4 in one day and 5 in
  another. Costs half the smart capacity for those hours, with the overload risk from 2026-10-01
  handled by the guard.
- **B. Rent a cloud GPU for the training runs only** (H100: ~2 h for E2B-sized). Fast, no prod
  risk, but it's the cloud-spend exception and the data leaves the box (it's all public datasets,
  plus teacher labels on public text).
- **C. CPU only for 0.8B.** Feasible but slow (likely 10+ h for a real run). Fine for smoke tests
  and zero-shot baselines, not for the ladder.

## Outputs

`BLOG.md` (stock 27B vs fine-tuned small vs Jev vs open replica, plus the batch-invariance
finding), and an HF release of the LoRA/merged weights and data manifest, with no TypeSafe data.
Nothing is pushed without Josh's go-ahead.
