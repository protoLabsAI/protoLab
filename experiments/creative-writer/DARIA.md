# Daria — the served artifact

Daria is the polished successor to `abliterated-de-slop-test-24b`: the **clean v6 direction**
plus the **gated projection-clamp**, served. It is not a checkpoint you can hand someone. The
clamp is nonlinear, so — unlike a constant control vector, which folds into a layer bias — it
**cannot be baked into weights**. Daria is weights + an inference-time intervention, and the
intervention has to ship as code.

    weights   /mnt/data/abliterate/mistral-creative-base-v0     (abliterated Mistral-Small-3.2-24B + voice SFT)
    direction /mnt/data/abliterate/creative-vectors/v6_mean_sentiment_dir.pt   (on-policy, unit-norm)
    clamp     h <- h - lam * relu(h.d_hat - tau) * d_hat        at layer 24, tau 0.41, lam 1.5

Why gated rather than constant: constant steering writes perturbed K/V that is re-attended
forever, so the perturbation compounds and long-form degenerates into verbatim loops by ~1000
tokens. The clamp's intervention is exactly zero once a token is at/below the boundary, so
nothing compounds. See `RESULTS.md`.

## Two ways to serve it

**vLLM (production).** `serve_daria_vllm.sh [port] [gpu]` — the clamp runs as a
`vllm.general_plugins` entry point (`daria_vllm/`, installed into `~/dev/vllm-025`) that hooks
the decoder layer inside the worker process. Real batching, streaming, paged attention.
Measured **262 tok/s aggregate at C=8** vs ~31 tok/s single-stream on the HF path. Config is
read from env at launch (`DARIA_ENABLE/DIR/LAYER/TAU/LAM`); it is fixed for the life of the
process. The clamp only touches the residual stream, so it composes with a quantized
checkpoint (NVFP4) unchanged.

**HF transformers (research).** `serve_daria.py` — single-locked `generate()`, ~31 tok/s, no
batching. Slow, but every knob is settable **per request** (`steer_lam`, `steer_tau`,
`steer_tau_end`, `steer_ramp_start`, `steer_ramp_start_tok`, `steer_ramp_end_tok`), which is
what makes it the harness for sweeps. `steer_lam: 0` disables the clamp, giving the unsteered
base from the same process — the cheapest honest control.

## Traps

Every one of these was hit for real. Each produces a plugin that loads, logs cheerfully, and
is either inert or subtly wrong. **Measure the output; never accept a log line as evidence.**

1. **Don't patch decoder-layer classes.** `MistralDecoderLayer` subclasses `LlamaDecoderLayer`
   and overrides both `__init__` and `forward`, so patching the Llama base class is a silent
   no-op for every Mistral model. The plugin printed "ACTIVE" and the first generation came
   back as pure unsteered slop. Fix: find the layer *module* after load and register a forward
   hook on it — no class-name guessing, works across architectures.

2. **vLLM 0.25.x has TWO model runners.** `vllm.v1.worker.gpu_model_runner` and
   `vllm.v1.worker.gpu.model_runner`, selected at runtime by `GPUWorker.use_v2_model_runner`;
   this build uses v2. Patching only v1 gives a plugin that loads, prints, and never runs.
   Patch every runner class that imports.

3. **vLLM's fused add+norm splits the residual stream.** Layers return
   `(hidden_states, residual)` and the actual stream is their **SUM**; HuggingFace returns the
   summed stream directly. Clamping `hidden_states` alone clamps the MLP delta, not the stream
   — numerically it differs from the reference by ~0.5 in activation units (the correct port
   matches to 2.4e-7). Clamp on the sum, apply the correction to `hidden_states`.

4. **tau assumes a unit-norm direction.** tau is a threshold on `h · d_hat`. Renormalising a
   non-unit direction silently rescales tau and every published operating point stops meaning
   what it says. The plugin refuses to run rather than quietly mis-steer.

5. **A fractional tau-ramp silently never fires.** If the ramp is expressed as a fraction of
   the caller's `max_tokens`, a caller that sends a generous budget defeats it: EQ-Bench sends
   `max_tokens=4096` for ~1200-token pieces, so a ramp starting at 0.70 would first engage at
   token 2867 — past the end of every piece. Use the absolute-token bounds
   (`--ramp-start-tok/--ramp-end-tok`) whenever the caller's budget is not the expected length.

6. **A concurrent A/B is not concurrent unless the job list interleaves.** `clamp_ab.py`
   originally built jobs arm-major (`for arm in lanes` outermost); `ThreadPoolExecutor`
   consumes FIFO, so one whole arm drained before the other began and the arms were
   time-separated — while the docstring claimed they shared GPU contention and wall clock.
   The fix is a loop reorder. Check the banked corpus, not the intent: the completion order
   of the records tells you what actually happened.

## The clamp earns its keep — settled 2026-08-19

Round 4 reduced the clamp's contribution to one claim: ~19% less deterministic slop on top of
`repetition_penalty 1.15`, and zero measurable craft. That claim rested on a single slop number
per arm. Resolved per run, the Round 4 data puts the delta at 1.4 sd with overlapping ranges —
**underpowered, not null**. A powered A/B (two concurrently-served lanes differing only in
`DARIA_ENABLE`, 9 runs x 32 prompts each, judge-free) settles it:

    clamp_off  slop 5.79 +/- 0.52      clamp_on  slop 4.29 +/- 0.61
    delta -1.50 per 1k words (-25.9%), exact permutation p = 0.00004114 (2/48620 — the
    minimum attainable at n=9/arm; all nine clamp_on runs sit below all nine clamp_off runs)

**Daria's honest spec: the craft and stability of `base + repetition_penalty 1.15`, with ~24%
less deterministic slop.** The clamp is a slop dial — that is the whole product claim, and it
is measured. The traps below are the price of it. See `RESULTS.md` Round 5.

## Known open items

- **Endings: SOLVED (2026-08-17).** A tail ramp fixes the artifact intrinsically, no system
  prompt needed — tau→0.0 or lam 1.5→4.0 over generated tokens 770–1100 both put Incongruent
  Ending Positivity *below base* (3.7 / 2.6 vs 4.2) and Overwrought at ~half base. Which of the
  two wins is undecided; the rubric numbers are n=1 and overlap.
- **Degeneration: SOLVED by the sampler, not the clamp (2026-08-18).** Four clamp geometries
  all degenerate 3-6 of 32 pieces (worst 4-grams 28-146); base is the only clean one. Adding
  **`repetition_penalty` 1.15** clears it outright: `tau0 + rep1.15` = **0/32, worst 4-gram 2**,
  cleaner than base's own worst of 8. Costs: Adherence 11.4->7.8, Coherent 13.8->11.2, pieces
  29% shorter. **Ship gate = worst-case per piece, never the mean** — medians are identical
  between a clean run and one with a 131x loop in it.
- **Judge coverage is a release gate too.** The bench judges with max_tokens=4096; a thinking
  judge burns that on reasoning and the piece goes unscored, so its reported rubric silently
  averages over 19-29 of 32 pieces depending on the run. Always re-judge with a real budget
  (`daria_judge.py`, 16k + reasoning_content fallback) before comparing runs.
- **The nonsense-refusal artifact: NOT clamp-attributable (2026-08-19).** `refusal_probe.py`,
  12 short prompts x 12 reps against clamp-on and clamp-off lanes, arms interleaved: **0/144
  in both arms**
  (95% CI 0-2.60%). That does not exclude the "low single-digit rate" originally reported —
  a 2.6% ceiling is consistent with 1-2% — but it rules the clamp out as the cause at any
  rate above ~2.6%. Separating a real 1-2% rate from zero needs ~1000+ calls per arm.
- **The ramp IS in the vLLM plugin, and is VERIFIED (2026-08-18).** Derived per token from the
  traced `positions` tensor. Verified by body-vs-tail slop, ramp-on vs ramp-off, on two
  concurrently-served lanes: body 3.94 vs 3.38 (unchanged, as predicted), tail 1.07 vs 5.18
  (-4.11); interaction -2.88 vs +1.80. A greedy diff is NOT a valid instrument here — vLLM
  greedy is nondeterministic run-to-run even within one server, so "identical" and "different"
  are both uninformative. Note the control also shows tails are naturally SLOPPIER than bodies
  (+1.80), independently reproducing the ending artifact the ramp exists to fix. n=9/8.
- **Lane is systemd-managed:** `daria-lane.service` (enabled, GPU0 :8045). ExecStartPost gates
  on a REAL completion plus proof the clamp hooked — /health returns 200 on a wedged engine,
  and a compile-cache hit serves the unsteered base while logging success.
