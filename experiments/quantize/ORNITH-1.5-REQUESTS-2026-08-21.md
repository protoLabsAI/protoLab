# Ornith-1.5 community requests — build report (2026-08-21)

Three hub requests landed within ~16 hours of the 1.5-9B GGUF release, all against files our
own card had listed as planned follow-ups:

| # | Repo / discussion | Ask | Status |
|---|---|---|---|
| 1 | `Ornith-1.5-9B-MTP-GGUF` #1 (juan-per-salv) | NVFP4 build | **BUILT + GATED** |
| 2 | `Ornith-1.5-9B-MTP-GGUF` #2 (rththr) | "iq4xxs or iq3 for 6 gb vram people" | **BUILT + GATED** |
| 3 | `Ornith-1.0-35B-NVFP4` #1 (klimekop6) | same treatment for 1.5-35B | **BUILT + GATED + IN PROD** |

Josh's call on #3: build it even though upstream `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` already
exists (it is what our own prod smart lane serves) — goodwill and a request from someone using
our work is sufficient reason. The card credits upstream rather than implying we are first.

## What shipped

**`Ornith-1.5-9B-NVFP4`** — 11.2 GB + 486 MB MTP sidecar, `/mnt/models/quantized/`.
128 LM linears W4A4; vision (333 tensors), DeltaNet, `lm_head`, MTP all bf16. Quantized from
our distilled MTP checkpoint rather than upstream bf16, so the release ships MTP-capable.
Gate: completion PASS / tool call PASS / vision PASS.

**Three IQ rungs** for the GGUF repo — IQ4_XS 5.45 GB, IQ3_M 4.67 GB, IQ2_M 3.87 GB, bundled,
MTP head pinned Q8_0. All fit a 6 GB card *with* the head.

## Findings worth keeping

**1. IQ4_XS beats Q4_K_M outright — the card's low-VRAM advice was wrong.**

    rung      size      base t/s   +MTP t/s   speedup   acceptance
    IQ4_XS    5.45 GB      228.4      276.9     1.21x        0.525
    IQ3_M     4.67 GB      233.8      257.0     1.10x        0.507
    IQ2_M     3.87 GB      260.7      276.8     1.06x        0.558
    Q4_K_M    5.78 GB      203.4      215.7     1.06x        0.544

Smaller, faster, bigger MTP gain. Q4_K_M was re-measured in the same session (203.4 vs the
206.7 already on the card) so the rows are comparable rather than cross-session.

**2. MTP does not invert on the i-quants.** The Q8_0 → Q4_K_M decay (1.77× → 1.06×) looked like
a trend heading for a regression at 3 and 2 bits. It isn't one: acceptance holds in a 0.51–0.56
band across the whole ladder and every rung is net-positive with the head on.

**3. The imatrix has no data for the MTP head — pin it.** A plain forward pass never activates
the `nextn` head, so all 15 `blk.32.*` tensors come back `did not find weights for…`. Unpinned
they would be i-quantized blind. `output.weight` and `token_embd.weight` are missing too and
fall back to defaults. Pinning the head to Q8_0 costs ~0.15 GB at Q4-class sizes.

**4. IQ2_M is genuinely usable.** Needle-exact at 4K/16K/32K, clean degeneration detectors, and
it solves a two-train word problem correctly *with a distance check* at 2.7 bpw.

**5. Calibration corpus deviates from house default — flagged, not hidden.** ThinkingCap's
imatrix rungs used Bartowski `calibration_datav3`; this ladder used a 70/30 mix of Ornith-1.5's
own generations and literary prose. The prose share is deliberate (instruct output is
register-narrow; creative writing degrades first at 3 bits), but the two corpora have **not**
been A/B'd. Do not compare i-quant rungs across the two releases without noting it.

**6. NVFP4 vision loss is not detectable at n=20** — and the n=5 read that suggested otherwise
was noise, exactly as [[feedback_underpowered_is_not_null]] predicts.

    probe                       bf16     NVFP4   p (Fisher)
    shapes                      20/20    20/20   1.00
    wordmark OCR exact           1/20     1/20   1.00
    wordmark OCR token correct    7/20    13/20   0.11

The wordmark row is the interesting one: **both precisions score 1/20**. The bf16 base itself
reads "protoLabs" as "protocolabs" and invents a trailing digit. Absolute OCR skill on a hard
glyph is a base-model property; only the *difference* says anything about the quant.

## What went wrong (harness, not artifacts)

**Token-starved probes produced three separate false failures.** Ornith-1.5 thinks adaptively,
so any budget under ~2K returns EMPTY content with `finish_reason=length`:
- the VL gate at `max_tokens=400` → "completion FAIL (0 chars)" on a healthy checkpoint;
- the GGUF bench at `n_predict=200` → all 24 sample texts empty, zero coherence evidence
  (speed numbers were unaffected — both arms decoded the same 200 tokens);
- and the gate also read `reasoning_content`, but vLLM 0.25 returns thinking in `reasoning`.

This is [[feedback_eval_prod_token_budget]], which we already had as a standing rule, tripped
three times in one session. The gate now defaults to `MAX_TOKENS = 4096`.

**The VL gate was mis-specified** and would have failed good checkpoints: it hard-gated on an
exact wordmark match that the *bf16 source* only hits 1/20. Rewritten — the hard gate is now
the shapes probe (is the vision path alive and correct, the thing quantization actually
breaks); the wordmark is informational; quant-vs-source verdicts go through `vision_parity.py`,
which tests the difference at adequate n.

**`verify_coherence.py` needs the lab venv** — system `python3` has no `openai`, and the script
fails with a traceback that a `set -e`-less runner will happily scroll past. First coherence
pass silently produced nothing.

## Files

    experiments/quantize/ornith15_9b_nvfp4_requant.py     9B NVFP4 (VL post-save fixups)
    experiments/quantize/ornith15_35b_nvfp4_requant.py    35B MoE NVFP4 — needs dual-GPU window
    experiments/quantize/gate_vl_quant.py                 completion + tool call + vision gate
    experiments/quantize/vision_parity.py                 quant-vs-source vision, Fisher exact
    experiments/quantize/imatrix/build_calibration.py     calibration corpus assembly
    experiments/quantize/imatrix/forge_iq_rungs.sh        IQ4_XS / IQ3_M / IQ2_M, head pinned
    experiments/quantize/imatrix/bench_iq_rungs.py        both arms per rung, acceptance
    experiments/quantize/imatrix/coherence_iq_rungs.sh    needle + detectors at depth
    models/serve-ornith15-9b-nvfp4.sh                     dense NVFP4 lane (MTP composes)

## SHIPPED (Josh-approved, 2026-08-21)

`protoLabsAI/Ornith-1.5-9B-MTP-GGUF` — IQ4_XS, IQ3_M, IQ2_M, **NVFP4**, fixed mmproj, new card.
`protoLabsAI/Ornith-1.5-9B-NVFP4` — new repo, 11.7 GB, vLLM W4A4 + distilled MTP sidecar.

**The NVFP4 GGUF rung had a hidden blocker worth recording:** `llama-quantize` has no NVFP4
target (absent from QUANT_OPTIONS even though `GGML_TYPE_NVFP4 = 40` and
`LLAMA_FTYPE_MOSTLY_NVFP4 = 39` both exist), **and gguf-py's NVFP4 class implements
`dequantize_blocks` only — there is no Python quantizer.** So neither the CLI nor the library
can write this format. The route is ggml's own `quantize_nvfp4` via ctypes on
`libggml-base.so` (`ggml_quantize_chunk`), which is the exact code path llama-quantize would
use. Validated by round-tripping the C output through gguf-py's independent Python
`dequantize`: corr 0.9955 NVFP4 / 0.999986 Q8_0 — two implementations agreeing on the block
layout, rather than trusting a hand-rolled format. Script: `imatrix/forge_nvfp4_gguf.py`.

**Final ladder** (one session, one box, one prompt mix, n-max 3, C=1):

    rung      size      base    +MTP    speedup  accept
    NVFP4     6.53 GB   216.1   299.1   1.38x    0.599   <- fastest outright
    IQ4_XS    5.45 GB   228.4   276.9   1.21x    0.525
    IQ2_M     3.87 GB   260.7   276.8   1.06x    0.558
    IQ3_M     4.67 GB   233.8   257.0   1.10x    0.507
    Q6_K      7.56 GB   171.6   249.7   1.46x    0.541
    Q8_0      9.79 GB   150.5   236.4   1.57x    0.543
    Q4_K_M    5.78 GB   203.4   215.7   1.06x    0.544

NVFP4 is fastest despite IQ4_XS and IQ2_M being *smaller* — FP4 sits on the tensor-core GEMM
path where MTP's verify is nearly free. Reproduces the 1.0 NVFP4xMTP finding on 1.5.
Note Q8_0 reads 1.57x here vs 1.77x in the card's older n-max sweep: different prompt mix,
acceptance 0.543 vs 0.663. Never compare across the two tables.

## The 35B — built, gated, and now the in-house smart lane

Quant ran in **13 minutes**, not the ~85 the Agents-A1 precedent predicted (41 sequential
stages @128 samples, peak ~90 GB/card). `run_35b_window.sh` stops prod with **restore in a
`trap`** — which earned its keep immediately: the first launch died in 8 s on a truncated
snapshot hash pasted out of a download log, and prod came back on its own. The requant script
now glob-resolves the snapshot instead.

**Ours is a different artifact from upstream's, which is the real justification:**

    ours      25.0 GB  compressed-tensors nvfp4-pack-quantized  ALL linear_attn bf16
    upstream  23.4 GB  ModelOpt 0.45 MIXED_PRECISION + FP8 KV   quantizes linear_attn.out_proj

Census: 30,720 packed expert tensors (40 x 256 x 3, exact), 160 packed attn, zero packed in
visual/linear_attn/mtp/lm_head/router, all 785 `mtp.*` preserved.

Gate: completion PASS / tool call PASS / vision 5/5 / wordmark OCR **3/3 exact** (the 9B scores
1/20 on that probe — it is capability-bound, not quant-bound) / **needle-exact at 200,409
prompt tokens**. Cut over at 23:03, verified on a real completion (not `/health`), then 89 GB
reclaimed: upstream's copy (22 GB) + the bf16 source (67 GB). /mnt/models 98% -> 88%.

**Carried debt, stated plainly: no scorecard was run before the swap** (Josh chose gate-only).
The claw 0.752 / FC 0.870 / LCB 0.365 figures in the serve-script header belong to UPSTREAM's
build; the header now says so. If lane quality is ever questioned, that is the gap.

**A grader bug found on the way, now fixed.** `verify_coherence.py` computed
`budget = min(max_tokens, max(256, 64512 - d))`, so every depth past ~62K got a **256-token**
generation budget — on an adaptive-thinking model that returns empty content, and the gate
printed "FAIL empty output" for a model that was needle-exact at 200K. Its own docstring warns
that a low cap "reads as empty output and false-fails the gate"; the auto-cap defeated it. It
also read `reasoning_content`, which vLLM 0.25 renamed to `reasoning`, so the fallback missed
too. Now takes `--ctx` / `--min-budget` and prints SKIP rather than FAIL when headroom is
genuinely insufficient. **That is the fourth time today this exact trap produced a false
failure** — it is worth treating token-starvation as the first hypothesis for any empty output
on this family.

## Measured scorecards (2026-08-22)

    build                          claw    reas    lcb     fc
    Ornith-1.5-9B-NVFP4            0.675   0.611   0.115   0.963  <- FC board-best
    Ornith-1.5-35B ours            0.719   0.861   0.205   0.889
    Ornith-1.5-35B upstream        0.752   0.861   0.192   0.870
    Qwen3.8-27B-NVFP4-MTP          0.761   0.778   0.632   0.870  <- prod, rolled back to

**The quants are faithful.** Ours vs upstream's 35B on the identical harness: LCB 0.205 vs
0.192, reasoning_hard identical at 0.861, claw within noise. No quantization penalty.

**But the model is weak at code, and prod was rolled back for it.** The Ornith-1.5 family
exhausts its token budget deliberating: 13/30 LCB problems on the 9B and the same signature
on the 35B. Thinking-on does NOT help — paired on identical problems it was worse (0.129 vs
0.329). Community reports on upstream's repos describe exactly this (failed one-shot HTML,
regression vs Ornith-1.0, context exhaustion). Since `coder` is one of the three aliases the
smart lane serves, the lane reverted to Qwen3.8-27B-NVFP4-MTP (LCB 0.632, 3x better).
Artifacts stay published and correct; they are just not the right prod lane.

## Open

- **35B**: needs `smart`/`reasoning`/`coder` down ~1.5–2 h. Host RAM (61 GB total, ~27 GB free)
  rules out CPU offload for a 72 GB bf16 MoE — it is a real dual-GPU window or nothing.
- **HF misreports two repos as `clip` / 0.46B params** (`Ornith-1.5-9B-MTP-GGUF` and
  `ThinkingCap-Qwen3.6-27B-MTP-GGUF`, the latter at 22k downloads) — HF's indexer picks the
  mmproj as the repo's representative GGUF. **Cause NOT determined**, and these were ruled out:
  alphabetical ordering (ggml-org/Qwen2.5-VL-7B-Instruct-GGUF has mmproj sorting first and
  reports correctly), commit order, mmproj `general.name` (ThinkingCap's was always correct),
  and stale index state (five fresh commits did not change it). The one workaround that would
  work -- no root-level mmproj -- is NOT safe: `find_best_sibling()` in llama.cpp's
  `download.cpp` requires the mmproj to share the model's directory, so moving it breaks
  `-hf` vision auto-download. **Next step is filing with HF, not more guessing.**
