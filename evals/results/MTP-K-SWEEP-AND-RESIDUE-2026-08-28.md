# Qwen3.8-27B-NVFP4: MTP draft-depth sweep + prefix-cache residue probe

**2026-08-28, replica B (:8042, GPU0). Prompted by [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)** —
same model, same `qwen3_5` GDN-hybrid arch, RTX 3090 / vLLM 0.27.1 instead of our
sm120 / 0.25.1 / NVFP4. Two of their findings were testable against our prod pair.

## 1. "Bug B" does NOT reproduce here — 128/128 clean

Their [gotcha 37](https://github.com/syv-ai/qwen38-27b-rtx3090/blob/main/docs/gotchas.md):
under a CAPTURED (FULL) verify step, a speculative request that **hits the prefix cache**
and whose prompt length lands on one residue mod 128 collapses — empty answer, or `"#"`,
or fluent invented content. Their fit is `R = 117 + k`, so k=3 predicts **residue 120**.

Our lanes are that config exactly: `cudagraph_mode=FULL_AND_PIECEWISE` (17 decode graphs
captured FULL), `enable_prefix_caching=True`, mamba cache mode `align`, mtp k=3.

**Result: all 128 residues, coverage 1.00, zero broken — including 120.**

```
residues swept   128 / 128        (pad steps 1 token, so every residue exactly once)
coverage         1.00 on all 128  (40-char window match against the source)
cache hit        28,800 of ~29,800 prompt tokens on every measured request
tok/step         3.99 of max 4.00
BROKEN           0
```

Both required conditions were reproduced, not assumed: each prompt was sent twice and the
second (a full self-hit, `cached_tokens` confirmed via `--enable-prompt-tokens-details`)
was the measurement. Scored on **coverage**, never on a failure signature — their own
hard-won point, since the same residue produced three different damage shapes for them.

Not a refutation of their finding, a scope limit on it: they saw it on 0.27.1 with the
lossy KVarN 4/2-bit cache. On 0.25.1 / sm120 / NVFP4 with an unquantized KV cache, it is
absent. **No action. Do not switch our lanes to PIECEWISE** — the mitigation costs
something and buys nothing here.

Harness: `evals/specdecode-residue/residue_sweep.py`, raw in
`evals/results/specdecode-residue/k3-full-8042.json`.

## 2. K=3 is CORRECT. The direct-to-lane result that said otherwise was a regime artifact.

**Conclusion reversed by the gateway measurement. K=3 stays.**

Direct to :8042 with **thinking OFF**, K=1 looked like a large win — replicated, tight,
two prompt lengths:

```
K   tok/step   C1 tok/s        C8 aggregate      (thinking OFF, direct to lane)
1     1.87     41-45           319-329
2     2.56     39.4            247.9
3     3.02     37.0            235-236
4     3.55     35.0            217.0
```

Through **ava:4000 / protolabs/smart at prod settings (thinking ON)**, it inverts:

```
K   C1 tok/s                  C8 aggregate                 (gateway, bg-matched)
3   36.8 / 36.8 / 36.7        234.0 / 232.4 / 226.7 / 251.7    (mean ~236)
1   21.0*                     136.4 · 185.3 · 137.9 · 125.0 · 147.7  (mean ~146)
```

**K=3 is ~+60% aggregate at C=8 through the gateway.** The K=3 arm is tight (227-252,
n=4) and the two ranges do not overlap. `served-by` was `protolabs/smart` on every
request — no cloud fallback masquerading as a local result.

\* the K=1 C1 cell was taken with 2.7 concurrent background requests on the lane; see below.

**Mechanism — thinking is what makes the deeper draft pay.** Reasoning text is long and
highly predictable, so acceptance is materially higher with thinking on, and that is
exactly the condition under which extra draft positions earn their step cost:

```
                              tok/step at K=3
thinking OFF (direct bench)        3.02
thinking ON  (live traffic)        3.11 - 3.44
```

Per-position acceptance under live thinking-ON traffic is 86% / 70% / 55%, and earlier
sustained traffic read 91% / 81% / 72%. Position 3 pays for itself there; with thinking
off it does not.

**Two methodology failures this exposed, both mine, both worth keeping:**

1. **The direct-to-lane bench disabled thinking.** That is not a smaller version of prod,
   it is a different workload, and it reverses the ranking of the knob under test.
   [[feedback_measure_through_the_gateway]] is not only about routing and fallbacks — it
   is about *request shape*.
2. **The first gateway A/B was confounded by uncontrolled live load.** Every K=1 run was
   taken after the single K=3 run, on a pair carrying real traffic, and C1 drifted 44.4 ->
   31.1 -> 26.3 -> 22.5 across replicates of the *same* config. Instrumenting lane
   occupancy showed one "single-stream" cell actually had 2.7 concurrent background
   requests. `gwbench.py` now samples `vllm:num_requests_running` on both lanes during
   every measured pass and prints `bg` with each cell; compare cells at matched `bg` only.

**Action: none. Both lanes remain at the unit default `SPEC_K=3`, drop-ins removed.**
K=2 and K=4 were only ever measured thinking-off and are not credible either; if anyone
revisits draft depth, sweep it through the gateway at prod settings from the start.

## 3. Two things confirmed, no action

- **`Failed to advance FSM` is noise, and there is a fatal sibling to alert on instead.**
  Their gotcha 40: under a speculator, tokens accepted past the grammar's termination point
  make 0.27.1 log `Unexpected: grammar rejected tokens ... Terminating request` and return
  an HTTP error for a valid request. Our 0.25.1 `backend_xgrammar.accept_tokens` has the
  same `_is_terminated` early-return-False, so the exposure exists — but our lanes show
  **688 and 66 noise lines, 0 fatal**. Our existing classification stands. Alert on
  `grammar rejected tokens`, never on `Failed to advance FSM`.
- **vLLM PR #50021 is unvendored here.** `mamba/ops/causal_conv1d.py:851` indexes
  `num_accepted_tokens - 1` with no bounds check; they hit illegal-memory-access with
  several concurrent MTP requests. We run 32. Zero occurrences so far — exposure, not a
  fire. Grep for `illegal memory access` if a lane ever dies unexplained.

## Not applicable to us

Both embedding matrices requantized, int8 Marlin activations + the negative-group-scale
sign bug, int4 KV, KVarN 4/2-bit cache, vision-tower CPU offload, sm80 repack, the whole
`CTX=long/huge` ladder. All of it buys VRAM we already have.

**DFlash2 stays off.** Their concurrency ladder independently reproduces our dFlash
finding: dflash2 wins C1-C4, MTP takes C8 (383 tok/s vs no steady state), because the
drafter runs out of recurrent-state pages at 5 residents. Our replica pair serves C=4-8
fan-out. A `syvai/Qwen3.8-27B-DFlash2-W4A16` drafter exists on HF if a genuinely
single-stream lane ever returns.

**One lever left on the table:** `--mamba-ssm-cache-dtype float16` (the flag exists in
0.25.1, `config/cache.py:135`, defaults `auto`). 48 of 64 layers are Gated DeltaNet with
fp32 recurrent state; they measured that state — not KV — as the concurrency bound, and
fp16 halved it with perplexity unchanged to three decimals. Our startup forces
`attention block size = 800 tokens` to match the mamba page size, so halving the state may
also shrink that block. Untested here.

## Incident during this work

Replica B was down ~40s: repeated restarts tripped `StartLimitBurst=3` /
`StartLimitIntervalSec=600` and systemd refused with `start-limit-hit`. Fixed with
`systemctl reset-failed` before restart — the trap already recorded in
[[feedback_orphan_kill_by_gpu_mem]]. **Any K sweep needs `reset-failed` between arms.**
The K=2 numbers were collected before the failure and are unaffected.
