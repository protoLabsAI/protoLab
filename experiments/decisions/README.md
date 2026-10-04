# decisions — Jev-style typed decisions on a stock vLLM lane

State in, typed answers with full probability distributions out: one forward pass per question,
no decoding, no reasoning channel. The server speaks TypeSafe's `POST /v1/systemone` wire format,
so the official `typesafe-sdk` can point at it. It runs against the prod smart lanes
(Swift-Qwen3.8-27B-NVFP4) with **zero training**.

Why: free-form generation on smart fails in two silent ways (protoLab#36). Replies can come back
blank (EOS inside `<think>`), and strict JSON can come back malformed (now fixed by the PR #44993
overlay). A typed decision call avoids both by construction, and it returns a probability the
caller can threshold and escalate on. It fills the gap between `experiments/jev/` (the
measurement study) and something a pipeline can call.

## Due diligence (2026-10-04)

- **TypeSafe Jev** (launched 2026-09-15): closed weights, cloud only. Primitives `noul` (P(yes)),
  `choice` (distribution over named options), `score` (distribution over ordered levels). Request
  shape `{state, model, questions: {id: {type, instructions, criteria}}}`.
- **[OpenJev](https://github.com/GitHub30/OpenJev)** (MIT): faithful copy of the wire schema and
  an HF-transformers in-process backend. **No vLLM backend**, so it can't use a production lane
  or share a state prefix across questions. That gap is what this fills.
- **[system-one-open](https://github.com/mithalouni/system-one-open)** (MIT): Gemma 4 E2B LoRA +
  Gemma 3 270M trained on 92 decision datasets, served on Modal. Single-token A–Z/a–z labels, 52
  options per pass. It reports, on TypeSafe's strict common subset (343 pairs): **Jev 86.9%**, its
  E2B 76.7%, stock Qwen 7B 73.8%. Its `typesafe_eval.py` parser is adapted here, with attribution.
- **TypeSafe's public eval** (`evals.typesafe.ai`): 4 workflows × 5 public cases = 20 cases, 372
  reference pairs. **The reference is the consensus of two frontier models** (gpt-6-astra,
  claude-fable-5-1), not human labels, and published Opus / Sol / TypeSafe answers sit beside it.
  So every number here is *agreement with frontier consensus*. TypeSafe's data is fetched by
  `build_typesafe.py` and **not redistributed**.
- vLLM gives what's needed natively: `/v1/completions` with `max_tokens: 1, logprobs: 20`, plus
  prefix caching. No server changes.

## Design (`s1serve.py`)

- Chat template with thinking **off**. Options labelled `A) name: description` (score:
  `0) level`). The model answers with the label. Labels are single tokens in Qwen3.x (A–T, 0–9
  verified), and vLLM returns the top 20 logprobs, so `choice` supports ≤ 20 options.
- Read the label tokens' probabilities and renormalise. `label_mass` (an extension field) is the
  raw mass on valid labels before renormalising: how in-schema the model was without a grammar.
  The median is 0.98.
- **State first, question last**, and all questions of a request go to one lane, so they share a
  prefix: N questions cost one long prefill plus N short ones. A 48-question invoice case (~30K
  characters of state) answers in ~6 s. A 3-question support ticket takes 0.3 s.
- `confidence` = top-1 probability. Per-kind temperature via `--temps` (not fitted yet).

```
~/dev/vllm-025/bin/python s1serve.py --port 8070     # backends default to :8041,:8042
python build_typesafe.py && python ts_eval.py --label <name>
```

## Results — TypeSafe public eval, strict common subset (343 pairs)

8 full runs (the variance is real, see below), against the published answers on the same pairs:

```
                    n    smart-s1 (8 runs)        opus    sol    jev
all kinds         343    0.766 (0.738-0.793)      0.907  0.915  0.875
  noul            214    0.876 (0.850-0.911)      0.967  0.949  0.930
  choice          103    0.580 (0.544-0.612)      0.835  0.883  0.825
  score            26    0.591 (0.538-0.692)      0.692  0.769  0.615
by workflow
  agent_trace      48    0.776 (0.688-0.833)                     0.729
  customer_svc     85    0.701 (0.635-0.800)                     0.882
  invoice         184    0.799 (0.783-0.815)                     0.924
  security         26    0.721 (0.692-0.769)                     0.769
8-run prob. average     0.784
```

- **A stock 27B with zero training, read in one forward pass, lands at 0.77**, between the
  trained open replica's E2B (0.767) and Jev (0.875), ~11 points behind Jev. It **beats Jev on
  agent traces** (0.776 vs 0.729).
- **The gap is `choice`** (0.58 vs 0.83). Wrong answers have high `label_mass` and the full
  option definitions are in the prompt, so it's not a harness artifact. These are multi-step,
  rule-heavy judgments ("compare like with like, ex-tax, net of permitted escalation…") that one
  pass without reasoning gets wrong, and that Jev was trained for. `noul` is close (0.88 vs 0.93).
- Calibration out of the box (per run): Brier ~0.13, ECE ~0.08 against agreement. No temperature
  fitted yet.

## The lane is not batch-invariant (a finding in its own right)

Identical prompts at temperature 0 give different answers depending on what else is in the batch.
That includes live prod traffic, which can't be controlled.

- **About 10% of answers flip between full runs** (39–40 of 372 on each lane separately). The
  worst single probability swing was 0.99.
- Controlled on one lane: sequential and concurrent requests are each stable in some invocations
  and unstable in others. Shared vs unique prefixes don't separate it. **It's batch composition,
  not prefix caching**, and it isn't GPU0 (A and B are equally unstable).
- Consequence: **report means and ranges across runs, never a single run.** Averaging
  probabilities over runs buys +1.8 points (0.784), the cheap ensemble.
- Untested: vLLM's batch-invariant mode (`VLLM_BATCH_INVARIANT`), whose NVFP4 + hybrid DeltaNet
  support is unknown, and whether this noise links to the #36 blank replies.

## Next

1. **Think-then-decide arm:** reason with thinking on, then do the one-token label readout after
   a closed `</think>`. That gives the accuracy ceiling for this model and the latency cost of
   reasoning. The readout also can't come back blank. This needs a quiet-traffic window
   (11Z–17Z): about 370 reasoning calls.
2. **Calibration:** per-kind temperature, leave-one-workflow-out, and AURC beside ECE (see
   `experiments/jev` rung 0: calibration doesn't transfer across families).
3. **Vera:** a decision layer over review findings (real? severity? verdict) with escalation below
   a threshold. The current labelled set (`evals/review-eval/truth.jsonl`, 26 rows) is too small
   to fit a threshold, so grow it first (#24).
4. **Publish:** blog draft (`BLOG.md`) and an HF artifact. The artifact would be the server plus
   eval harness, a rebuild script instead of TypeSafe's data, and the per-run results.
   Nothing ships without Josh's go-ahead.
