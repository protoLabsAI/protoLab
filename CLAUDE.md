# CLAUDE.md — protoLab (the heavy rig)

This repo is **substrate #2 — Models** in the [protoLabs.studio](https://protolabs.studio) portfolio. Heavy rig: 2× RTX PRO 6000 Blackwell, 192 GB VRAM total. FP8 / vLLM at scale. The model gateway every other studio service hits.

Sibling: [`avaLab`](https://github.com/protoLabsAI/avaLab) on the A6000 prosumer rig (GGUF / llama.cpp / Ollama / ComfyUI). Cross-cutting work lives here.

Studio-wide brand contract: [`protoLabsAI/studio-brand`](https://github.com/protoLabsAI/studio-brand). Read [`docs/explanation/portfolio.md`](https://github.com/protoLabsAI/studio-brand/blob/main/docs/explanation/portfolio.md) before reshaping anything in this repo's structure.

## Audience filter

The §3 filter from `studio-brand/docs/reference/foundation.md` applies to everything we publish from here:

- If we have to explain context or tokens — not our audience.
- If they need more than quickstart + example code + docs — not our audience.
- If they can't read the experience via CLI output and raw JSON — not our audience.

Open-source code itself has no audience filter — anyone can fork. The breakdowns, README copy, model cards, and dataset cards we publish *do*. No intro-to content. No hand-holding. **Assume competence.**

## What this repo ships to the brand

- **Findings.** CUDA graphs on Blackwell (+37–470%), INT4 routing corruption on MoE, FP8 KV cache broken on sm120, NCCL_P2P_DISABLE=1 fixing TP=2 PCIe — patterns to study and steal.
- **Models on HF.** FP8 quants of Qwen3.6 family published to [`protoLabsAI`](https://huggingface.co/protoLabsAI) via `experiments/quantize/`.
- **Eval suite.** `evals/` is the lab's open-source pattern. claw-eval is already a public substrate.
- **The gateway.** `infra/gateway/` is operational infrastructure for every other studio service.

## What was parked 2026-05-22

ORBIS retired as a product. Coding-agent and prompt-product work out of scope. Side bets parked under `experiments/*/PARKED.md`: companion-stack, voice-agent, salm-duplex, stt-whisper, tts-compare, agent-lightning, rlm, qwen3-omni, image-gen-eval, flux2, pixel-gen, ltx-video. Audio-tags survives as the standalone brand exemplar.

The thesis "small specialized models per pipe of the conversational loop" survives in audio-tags. If a new consumer surface appears, unpark one pipe at a time as a fresh experiment.

## Structure

- `packages/lab-core/` — Pydantic models, GPU utils, path constants. Publishable.
- `evals/` — claw-eval (submodule), custom suites, function-call, RAG, refusal. Strict + tested.
- `models/` — vLLM configs (`vllm-swap.sh`), MoE kernels (`moe-configs/`), benchmark scripts.
- `training/` — fine-tuning workspace (LLaMA-Factory configs, TRL).
- `experiments/` — active: `audio-tags/`, `context-1/`, `quantize/`, `embedding-bench/`, `gemma4-eval/`, `proto-bench/`, `rag-bench/`, `vllm-dashboard/`. Everything else parked.
- `infra/` — gateway (LiteLLM + Langfuse), vLLM systemd, Prometheus exporters.

## How we operate

The cycle:

```
experiment → report → engineering → test → content → repeat
```

Each phase has an exit criterion; don't move on until current phase is done.

| Phase | What | Exit when |
|---|---|---|
| **experiment** | research in `experiments/<name>/` | model artifact + tier-0 baselines + cross-domain held-out eval |
| **report** | internal `RESULTS.md` with honest numbers and what didn't work | written without softening — for next-session-us |
| **engineering** | wire the artifact into a consuming surface (gateway, eval, training) | PR landed on consuming repo's main |
| **test** | validate under real conditions | one round of real-world signal (real users, real traffic) |
| **content** | public artifact: HF model + dataset cards + protolabs.studio blog post | merged blog + public HF release |
| **repeat** | next experiment, informed by what shipped | always |

Default to publishing publicly via `protoLabsAI/` on HuggingFace and protolabs.studio for the writeup. Privacy is a drafting state, not a target. **Every shipped experiment produces a blog draft in `experiments/<name>/BLOG.md` before the next experiment starts.** audio-tags is the template.

## Daily setup (dual GPU) — 2026-08-23 (pm): Qwen3.8-27B REPLICA PAIR, one per card (CANONICAL)

**★ CURRENT PROD (2026-08-23 pm).** **Two IDENTICAL Qwen3.8 lanes, one per card**, no TP —
the fleet is now a replica pair, not a lane-per-model split. Both cards are prod cards.

| GPU | Service | Model | Port | util | seqs | KV pool | Notes |
|-----|---------|-------|------|------|------|---------|-------|
| 1 | `vllm-smart-qwen38.service` | Qwen3.8-27B-NVFP4 + MTP K=3 | :8041 | 0.86 | **32** | **771,146 tok** | `smart reasoning coder qwen3.8-27b-nvfp4` |
| 0 | `vllm-smart-qwen38-b.service` | Qwen3.8-27B-NVFP4 + MTP K=3 | :8042 | 0.86 | **32** | **770,382 tok** | **replica B** — same served names, same everything |
| 1 | `embed-server.service` | Qwen3-Embedding-0.6B | :8001 | — | — | — | embed A |
| 0 | `embed-b.service` | Qwen3-Embedding-0.6B | :8004 | — | — | — | embed B |
| 0 | ComfyUI | LTX pipelines | :8188 | — | — | — | **on demand only** — see below |

Measured after cutover: GPU0 83.3/95.6 GiB (11.7 free), GPU1 84.6/95.6 GiB (10.4 free).

**Why: the single lane was saturated, and it was measured, not assumed.** Before the change,
`:8041` sat at **16 running (pinned at its `--max-num-seqs` cap) with 11 queued on
`reason=capacity`**, KV 87.5%, 18 preemptions, 982 requests in 13.6 h. Two levers were
pulled together:

1. **`--max-num-seqs` 16 → 32 on both lanes.** The A/B in `evals/results/maxseqs-ab/`
   predicted C=16 TTFT p50 ~2297 ms → ~376 ms (~6×) and agg 343 → 397 tok/s, costing ~60%
   TPOT at C=8. **Confirmed in the field**: the queue drained to *6 running / 0 waiting*
   immediately after restart. This was affordable only because util 0.86 had already put the
   KV pool at ~774k tokens.
2. **A second, identical lane on GPU0.**

**What paid for GPU0 — both tenants stopped AND disabled.** GPU0 had 19.1 GiB free against
~26.7 GiB of weights, so it could not host a replica as configured:

```
lane                  VRAM held   requests served in 13.6h
vllm-protopen (:8050)   52.5 GiB          12
daria-lane    (:8045)   21.6 GiB           6
vllm-smart    (:8041)   84.6 GiB         982   <- saturated
```

**The `disable` is load-bearing — same rule as Fish TTS, do not just `stop` them.** Left
`enabled` they return at boot, race `:8042` for GPU0, and one OOMs. **Reviving either means
dropping `:8042`'s util FIRST.** `protolabs/creative` (daria) has **NO FALLBACK by design**,
so it now fails loud rather than degrading — that is intended, don't "fix" it with a fallback.

**Keep the pair in lockstep.** The two units differ **only** in `GPU` and `PORT` and both run
`models/serve-qwen38-27b.sh`. The gateway load-balances them with `routing_strategy:
least-busy`, which picks a lane per request — so any drift between them (util, seqs, effort
pin, token clamps) becomes a coin-flip in behaviour that no caller can see or reproduce.
Change both or neither.

**GPU0 no longer needs the serialization drop-in** while `:8042` is its only vLLM lane. If
protopen/daria are ever re-enabled, **restore the `After=` ordering first** — the
never-parallel-start-same-GPU-lanes invariant still holds.

**Gateway side is NOT done on this node** — it lives on ava. Filed as
[homelab-iac#250](https://github.com/protoLabsAI/homelab-iac/issues/250) with the exact YAML;
a ready-to-apply patch is at `infra/gateway/patches/2026-08-23-smart-replica-b-8042.patch`.
Five groups name `:8041` (`smart`, `reasoning`, `coder`, `vision`, `fast`) and each needs a
twin deployment on `:8042`; a duplicate `model_name` is a LiteLLM **load-balanced pool**, not
a fallback. **Until that lands, `:8042` receives zero traffic.** Note the compose file's
single-FILE mount inode trap: deploy with `docker compose up -d --force-recreate gateway`,
never a plain `restart`.

**Found while wiring this: `embed-b` (:8004) gets 0 traffic because it has NO deployment
entry in the gateway config at all** — only `:8001` is listed. This corrects the earlier
note in this file that called it a load-balancing failure; the gateway was never told the
lane exists. Same one-line fix, folded into #250.

**Rollback to the 3-lane split:** `systemctl disable --now vllm-smart-qwen38-b`, then
`systemctl enable --now daria-lane vllm-protopen` (daria first — protopen's drop-in gates on
it). Primary unit backup:
`~/dev/.vllm-bump-review/unit-backups/vllm-smart-qwen38.service.pre-seqs32-20260823`.

**Fish TTS is stopped AND disabled — do not just `stop` it, the `disable` is load-bearing.**
At smart util 0.86 the card has 10.4 GiB free; Fish wants 19.4 GiB. Left `enabled` it comes
back at boot, loses, and takes the prod lane down with it. To bring Fish back you must first
drop smart's util (0.86 → ~0.62) — the two cannot coexist as configured.

**ComfyUI is a swap-in, not a co-tenant (Josh, 2026-08-23).** It sits on GPU0 holding ~0.5 GiB
idle, but LTX renders want ~37 GiB — they will OOM. **This got STRICTER with replica B**: GPU0
now leaves only ~11.7 GiB free, and there is no longer a fat protopen lane to shed. The only
GPU0 lane to stop is `vllm-smart-qwen38-b` itself (frees ~83 GiB), which halves smart capacity
for the duration — so schedule renders, don't improvise them. `POST /free` after. The unit is
`disabled` at boot, which is correct; keep it that way.

**GPU0 serialization is DORMANT, not deleted (updated 2026-08-23 pm).** With daria and
protopen disabled, `vllm-smart-qwen38-b` is the only vLLM lane on GPU0 and nothing races it.
The drop-in `/etc/systemd/system/vllm-protopen.service.d/10-gpu0-serialize.conf`
(`After=daria-lane.service`) is still on disk and still correct — **it just has no work to do
while both those units are disabled.** The invariant it encodes is unchanged and applies the
moment a second GPU0 lane returns: every lane profiles free memory at startup, so two started
in parallel both size KV against a nearly-empty card and one OOMs. **Never parallel-start
same-GPU lanes** — if you re-enable protopen/daria, order them against `:8042` as well, not
just against each other.

**protopen's serve scripts were missing on this branch — now tracked (2026-08-23).**
`models/serve-protopen-nvfp4.sh` + `models/protopen_health.sh` existed only in commit
`4802692` on another branch, so the unit's `ExecStart` pointed at files that did not exist
on `quant/ornith-1.5-requests` and the lane could not start. Restored from that commit and
committed here. **Lesson: a systemd unit can be `enabled` and reference an `ExecStart` that
no longer exists on the checked-out branch — `systemctl is-enabled` says nothing about
whether the lane can actually start.** Worth checking whenever a lane is "temporarily" down.

**Measured 2026-08-23 — smart lane at util 0.86, coherent prompts, 0 preemptions everywhere.**
Random-token load is banned here: it defeats spec decode (MTP acceptance collapses on
unpredictable tokens), so a `--dataset-name random` sweep understates this lane badly.

```
input   C    ttft p50   ttft p99   tpot p50   agg tok/s
short   1        870ms      870ms     30.2ms       28.3
short   8        183ms      184ms     31.3ms      221.8
short  16        291ms      293ms     35.6ms      392.2
~2.8k   8       1400ms     1460ms     34.5ms      152.6
~2.8k  16       2104ms     2409ms     43.4ms      223.3
```

Peak VRAM under load was 86.7 GiB — only **1.8 GiB of transient activation above idle**, so
the 10.4 GiB cushion is real headroom. Note this validates the util raise *under load*, not
at boot: `--gpu-memory-utilization` IS profiled (unlike `--kv-cache-memory`, the 2026-08-11
trap), but a clean startup still proves nothing on its own.

**The util raise removed KV as a constraint; it did NOT fix the C=16 TTFT ceiling.** At
multi-k prompts C=16 still sits at ~2.1 s p50, and that is a **`--max-num-seqs` artifact, not
a KV one**. The lane runs `--max-num-seqs 16` and the A/B in commit `4802692`
(`evals/results/maxseqs-ab/`, in 1k / out 256) measures the tradeoff directly:

```
max-num-seqs   C=8 ttft p50   C=8 tpot p50   C=16 ttft p50   C=16 agg tok/s
16                    ~286ms         25.6ms         ~2297ms            ~343
32                    ~311ms         41.5ms          ~376ms            ~397
```

seqs32 is ~6× better on TTFT at C=16 and scales on to C=24/32 (485/573 tok/s), but costs
~60% on TPOT at C=8. Previously seqs32's KV was unaffordable; **at 774k tokens it now is** —
raise MAXSEQS, not util, when the C=16 tail starts hurting. Left at 16 deliberately so the
util change isn't confounded; traffic is currently light (514 requests/24h).

**Rollback:** unit backups in `~/dev/.vllm-bump-review/unit-backups/`
(`vllm-smart-qwen38.service.pre-util086-*`, `daria-lane.service.pre-seqs1-*`).

**Driver is 610.43.02 / CUDA UMD 13.3** (was documented as 595.45.04 — corrected 2026-08-23).
The GSP-hang note in [[reference_gsp_hang_blackwell]] is pinned to the 595.x-open series and
may no longer apply.

**Probe hygiene, re-learned 2026-08-23:** a `max_tokens: 16` "say ok" probe returns **empty
content** on both smart and protopen — `finish_reason: length`, all 16 tokens spent in
`reasoning`. The lanes were fine; the probe was token-starved. Same trap as
[[feedback_eval_prod_token_budget]]. Gate with `enable_thinking:false` **and** a real budget,
and always print `finish_reason` + `reasoning` before calling a lane broken.

> **Below: the 2026-08-10 DSV4-Flash jasl lane** — SUPERSEDED 2026-08-15 by Qwen3.8-27B
> (`vllm-smart.service` is `disabled`, `vllm-smart-qwen38.service` took :8041). Kept for the
> jasl-fork build notes, the KV/`--kv-cache-memory` traps, and the TP=2 tenancy rule.

## [SUPERSEDED 2026-08-15] Daily setup (dual GPU) — 2026-08-10: DSV4-Flash smart lane on the jasl fork

**[EX-PROD 2026-08-10 → 2026-08-15].** `vllm-smart.service` (now `disabled`) ran
`models/serve-dsv4-flash-jasl.sh`: DeepSeek-V4-Flash-0731 on the **jasl vLLM fork**
(pin `aa0d513027`, env `~/dev/vllm-jasl`, rebuild via `models/build-vllm-jasl.sh`) —
TP=2 both GPUs :8041, serves `smart reasoning coder deepseek-v4-flash`, embeds coexist.

| Change vs stock 0.25.1 lane | Value |
|---|---|
| Decode C1 (real prompts) | **184 t/s vs 103.5 (+78%)** — DSpark spec decode K=5 (K≥5 REQUIRED, block size) |
| Decode C8 | neutral (−0.6%) — NOT the dFlash inversion |
| Context | **393216 (384K)** — DeepGEMM >256K ceiling fixed in fork indexer; `reasoning_effort: max` now legal |
| KV pool | **689,133 tok (1.75× @ 384K)** via `--kv-cache-memory 8589934592`, fp8 KV — see the 2026-08-11 correction below; the 11859195904 / 951,437-tok figure measured on 08-10 runtime-OOMs and must NOT be restored |
| Scorecard | LCB 0.633↑ FC 0.907↑ claw 0.744 · reasoning_hard 0.861 (−0.11, thinking-LENGTH delta — see below) |

Full characterization: `evals/results/DSV4-JASL-TEST-2026-08-10.md`. KV footnote: stock's
"1.89M tokens" was a reporting bug (2.19× over-report); real stock capacity ≈ 860K.

**xgrammar "Failed to advance FSM" errors are LOG NOISE, not corrupted output (measured
2026-08-11).** The jasl lane logs `backend_xgrammar.py:162 Failed to advance FSM for request
… for tokens N` on guided-decoding requests. Root cause is **adaptive thinking × guided
decoding** — clean A/B, identical schema:

```
                   FSM errors   reasoning chars   schema-valid
thinking ON  (12)      18          3335 mean         12/12
thinking OFF (12)       0             0              12/12
tools    ON  (10)       5             —              10/10
tools    OFF (10)       0             —              10/10
```

xgrammar is fed reasoning tokens that can't match the JSON grammar and complains; the engine
applies the grammar only to the content stream, so **output integrity is unaffected**.
**86 guided/tool requests across 7 schema shapes (flat, nested, array-of-objects, enum,
regex, 4-deep, anyOf) — 100% schema-valid, zero violations.** structured 0.917 / FC 0.907 are
NOT suspect. ⚠️ But **do not alert on this string**: every thinking+guided request emits
~0.5–1.5 lines, so a *real* FSM failure would be buried. Fix (or upstream report) first.
Lesson: this was initially escalated to top priority on log severity alone — measure the
output before ranking a defect.

**Known deltas / open items:**
- **Default thinking = adaptive-ON** (short reasoning on every request in `reasoning`
  field) vs stock's default-OFF. `thinking:false` / `reasoning_effort:"none"` force off.
  Gateway aliases (homelab-iac#215) still assume default-OFF — **pin per-alias on ava**.
- **reasoning_hard −0.11**: fork deliberates ~2/3 as long on hard tasks. Mitigation =
  `reasoning_effort: xhigh/max` (2.4× deliberation, 384K allows max) — **unproven on the
  suite**; add an effort knob to `runners/run_custom` and re-run.
- **Accepted debt: no supply-chain review** of the fork (source build from github.com/jasl/vllm).
- Rollback: `unit-backups/vllm-smart.service.pre-jasl-20260810-*` (stock 0.25.1 lane, ~2 min).
- OS drive at 99% (~11G free) after the env build — Windows NVMe reclaim (~1TB) is the fix.

**Vision lane RETIRED from this node 2026-08-11 (same day it landed) — moving to ava.**
`vllm-vision.service` stopped + **disabled**; `embed-b.service` re-enabled (GPU0 :8004);
smart lane KVMEM raised **5637144576 → 8589934592** (8 GiB; KV pool 452,238 → **689,133
tokens**, max concurrency 1.15× → **1.75×** @ 384K). KV dtype stays **fp8** (unchanged).
Validated under load, not just at boot: C=8 burst on 20K-token prompts (13 concurrent with
live traffic) → 8/8 clean, **0 preemptions**, ~3.6 GiB free per card.

**Do NOT restore 11859195904 here** (the 2026-08-10 "fully utilize" value) — tried it first
and it FAILED: the pool allocates fine (951,437 tok, 2.42×) and then dies on the first real
inference with `torch.OutOfMemoryError` on **GPU1**, wanting 256 MiB of transient activation
with ~245 MiB free. Vision only ever constrained GPU0; **GPU1's tenancy (embed-A, 1.81 GiB)
never changed**, so GPU1 could not hold an 11.3 GiB pool with or without vision. Yesterday's
walk-down in the log (603,011 → 516,890 → 452,238) was the same wall being hit from the
GPU0 side.

**Two traps this exposed, both reusable:**
1. **`--kv-cache-memory` is NOT profiled.** vLLM allocates exactly the bytes you ask for and
   only discovers the shortfall when real activations land. A clean startup proves nothing —
   always leave runtime headroom and validate under load, not at boot.
2. **`/health` lies when the engine wedges.** After the worker OOM'd, the API server kept
   returning **200** while the engine core sat behind a dead worker (`No available shared
   memory broadcast block found in 60 seconds`) serving nothing. Any systemd gate or gateway
   health check would have kept routing traffic to it. **Gate on a real completion.**

**Why it was pulled:** the one-day-old vision lane cost the smart lane more than it was
worth. Under real fan-out DSV4 was **KV-starved** — 13 running / 49 queued (44 blocked on
`capacity`), KV usage 91%, **mean TTFT 50.5 s** — while vision had served **7 requests /
646 prompt tokens** in ~4.9 h of uptime. Decode was never the problem (14.8 ms TPOT ≈ 67
tok/s/stream, **0 preemptions**); the queue was.

**The TP=2 tenancy rule this taught us — worth keeping.** Under TP=2 the KV pool is sharded
symmetrically, so it is bounded by the **tighter card**. A lopsided tenant strands headroom
on the other one: with vision (7.3 G) on GPU0 and embed-A (1.9 G) on GPU1, GPU0 had 688 MiB
free while GPU1 sat on 6128 MiB that **no request could ever use**. Keep co-tenants balanced
across both cards, or the imbalance is paid twice. (Consolidating both tenants onto one card
is *worse*, not better — it just moves the binding constraint.)

**Note `--max-model-len` is NOT a KV lever here.** With `--kv-cache-memory` pinned, the pool
is a fixed byte allocation; lowering max-model-len frees ~0 bytes (activation memory scales
with `--max-num-batched-tokens`, not context length). It only lowers the *floor* vLLM will
accept (5.01 GiB for one 393216-token request) and the reported concurrency ratio. Cutting
context to buy KV does not work — buy bytes instead. 384K also stays load-bearing: it's what
makes `reasoning_effort: max` legal.

**MEASURED 2026-08-11 — full `speed-test-v2 full` sweep at KVMEM 8 GiB, 25 cells, 0 errors.**
Full table + caveats: `evals/results/speed-v2/dsv4-jasl-8gib-20260811-104306/RESULTS.md`.

```
regime      C  ttft p50  tpot p50  agg tok/s  goodput      regimes: chat 1k/1k
chat       16     302ms    29.3ms      508.4     0.50       context 8k/1k
context     8    1343ms    30.2ms      248.0     0.22       gen 1k/8k
context    16    1405ms    49.0ms      309.8     0.12       think 1k/16k
think      16     320ms    12.2ms     1096.0     0.06       legacy 128/800
legacy     16     182ms    28.0ms      529.2     0.66
```

**`--max-num-seqs 16` is a HARD CEILING — the top actionable finding.** The C=32 cliff
reproduces in **all five regimes**: throughput flat (−2% to +13%) while TTFT p50 collapses
to 21–216 s and goodput goes to ~0. `think` C=32 has a **216 s median** TTFT. Above 16
concurrent, requests queue instead of degrading. The lane **already runs at the cap** (13–15
concurrent under load; 13 running / 49 waiting in the KV incident) and KV can now afford more
(44% util, 0 preemptions) — raise MAXSEQS, not KVMEM.

**Optimal concurrency is C=16 — except long prompts, where it's C=8.** `context` (8k) pushes
TPOT to 49.0 ms p50 / 62.9 p99 at C=16, through the 50 ms SLO; goodput peaks at C=8 (0.22)
and falls to 0.12 at C=16. Long-output work is this lane's strength: `think` does 1096 tok/s
at C=16 with 320 ms TTFT, and 3.6 ms TPOT (~278 tok/s/stream) at C=1.

⚠️ **`--dataset-name random` silently defeats spec decode — a caveat the script's honesty
notes don't mention.** A draft head can't predict random tokens: DSpark acceptance measured
**11–22% (len 1.57–2.12)** during this sweep vs **48.9% (len 3.44)** on coherent prompts. The
fork's headline feature contributes ~nothing here while still paying 5–10× verify-batch
inflation. **These numbers are a floor**, and benchmarking this lane against a
non-spec-decode lane on random data is actively unfair to it. (Same day, live traffic
measured 1.47 s mean TTFT — better than `context` predicts, via cache hits + coherent text.)

**Still on the table (not taken — needs a load-test window, not a boot check):**
- **KVMEM 8 → ~10 GiB.** ~3.6 GiB/card was left as cushion; the true ceiling is between
  8 GiB (holds) and 11.05 GiB (runtime-OOMs). Probe with `speed-test-v2.sh full` at real
  prompt sizes, not a single request.
- **`--max-num-batched-tokens 4096` is the bigger TTFT lever, and it is untouched.** In the
  KV-starved incident only **8.6 s of the 50.5 s mean TTFT was queue wait** — the other ~42 s
  was prefill. At MNBT 4096 a p88-sized 50K-token prompt needs 12+ chunked-prefill iterations
  competing with decode. Raising MNBT costs activation memory (which is why it pairs *against*
  a KVMEM raise — they draw on the same cushion). Decode was never the problem: 14.8 ms TPOT
  ≈ 67 tok/s/stream throughout.

**Consumers left dangling until MiniCPM lands on ava:** `protolabs/vision` gateway alias
(dead :8050) and protoAgent `knowledge.image_describe_model: protolabs/vision` in
`~/.protoagent/default/config/langgraph-config.yaml`. No local vision fallback — the fast
lane (Ornith-35B-NVFP4), which *out-visions* MiniCPM (OCRBench 89.7 vs 77.3, MMMU 75.6 vs
50.2), has been down since the 2026-08-08 DSV4 cutover. To bring vision back here instead:
re-enable `vllm-vision.service`, disable embed-b, drop KVMEM to 5637144576.

> **Below: the prior 2026-07-22 setup** — kept for Ornith/Laguna serving notes and rollback context.

## [SUPERSEDED 2026-08-10] Daily setup (dual GPU) — 2026-07-22: vLLM 0.25.0 default, Ornith fast + Laguna-S smart

**[EX-PROD 2026-07-22].** vLLM **0.25.0** (`~/dev/vllm-025`) is now the default env — cut over from
0.24.0/`vllm-024-test` (which is untouched as rollback). torch stays 2.11.0+cu130, flashinfer 0.6.13, sm120
recipe unchanged. Behavior-preserving on Ornith (206 tok/s, tools clean).

| GPU | Service | Model | Port | Notes |
|-----|---------|-------|------|-------|
| 0 | `vllm-fast.service` (→vllm-025) | Ornith-1.0-35B-NVFP4 | :8040 | `fast` — orchestrator/agentic, `qwen3_xml` tools, fp8-KV, util 0.33 |
| 1 | manual serve (vllm-025) | **poolside Laguna-S-2.1-NVFP4** | :8041 | `smart` — 118B/8B agentic coder (SWE-bench 78.5%), `poolside_v1` parsers |

- **Laguna needs vLLM 0.25.0** (0.24.0 garbles tools/multi-turn — every band-aid failed; the version IS the
  fix). Serve: `--trust-remote-code --reasoning-parser poolside_v1 --enable-auto-tool-choice --tool-call-parser
  poolside_v1 --override-generation-config '{"temperature":0.7,"top_p":0.95}' --moe-backend marlin
  --kv-cache-dtype fp8`, `HF_HUB_OFFLINE=1`, sm120 env. The override is load-bearing (raw sampling defaults
  degrade NVFP4). Full playbook: [[reference_laguna_serving]].
- **S is a manual serve** (not yet a systemd unit — make one when it graduates from vibe-testing).
- **Rollback fast to 0.24.0**: restore `/etc/systemd/system/vllm-fast.service.pre-025-bak` + daemon-reload + restart.
- Media (ComfyUI/Fish/embed on GPU1) currently displaced by Laguna-S; bring back if S moves off GPU1.

> **Below: the prior 2026-07-20 2-lane (gemma-31B smart on 0.24.0)** — kept for the gemma-4 landmines,
> sequential-start rationale, and rollback path.

**[prior] CURRENT PROD.** Consolidated from the 3-lane fleet after the Gemma-4 v2 re-pass
(`evals/results/GEMMA4-REPASS-2026-07-19.md`): **GPU0 = LLM card, GPU1 = media card.**

| GPU | Service | Model | Port | util | Notes |
|-----|---------|-------|------|------|-------|
| 0 | `vllm-fast.service` | Ornith-1.0-35B-NVFP4 | :8040 | 0.33 | `fast` — orchestrator/agentic/vision, tools (`qwen3_xml`), 256K, fp8-KV (unchanged) |
| 0 | `vllm-smart.service` | **RedHatAI/gemma-4-31B-it-NVFP4 + MTP K=1** | :8041 | 0.59 | serves `smart` + legacy `reasoning`/`coder` aliases; gemma4 parsers, non-thinking, 256K, fp8-KV, `--language-model-only` |
| 0 | `embed-b.service` | Qwen3-Embedding-0.6B | :8004 | — | embed B |
| 1 | ComfyUI | LTX-2.3 NVFP4 / Krea / ACE-Step / Ideogram | :8188 | — | media card — full headroom for video/image/music |
| 1 | `protovoice-stack.service` | Fish S2-Pro TTS | :8092 | — | lazy ~20G (inactive unless needed) |
| 1 | `embed-server.service` | Qwen3-Embedding-0.6B | :8001 | — | embed A |

- **Why 31B**: board-best LCB (0.708 > CoderNext 0.645), board-best structured_hard (0.849),
  IF 0.90, FC 0.926, claw 0.724 — one dense 22G checkpoint ≈ the reasoning+coder lanes.
  ThinkingCap keeps a reasoning_hard edge (0.889 vs 0.806) — rollback below if that matters.
- **Sequential start still REQUIRED on GPU0**: smart gates on fast `/health` (ExecStartPre).
  Restart order: fast → smart → embed-b.
- **Gemma-4 serving landmines**: nvidia ModelOpt NVFP4 does NOT load on 0.24.0 (tie_weights
  NotImplementedError) — compressed-tensors only. Forced `enable_thinking:true` is broken on
  the family (12B: parser swallows the entire output) — the lane is non-thinking (gemma4 structural
  default), which is also the correct coding config. TRITON_ATTN env required.
- **Rollback to 3-lane**: `vllm-reasoning.service` (ThinkingCap :8041) + `vllm-coder.service`
  (Coder-Next :8032) on disk, **disabled** 2026-07-20; backups
  `~/dev/.vllm-bump-review/unit-backups/*.pre-2lane-20260720-*`. Stop smart → enable+start
  reasoning (then coder on GPU1, which again competes with ComfyUI for VRAM).
- **Gateway**: ava-side remap needed — `protolabs/coder` base_url :8032→:8041 (model name
  `coder` still served); `protolabs/reasoning`/:8041 unchanged; new alias `protolabs/smart`
  recommended. Until remapped, coder alias 404s (lane was already stopped pre-cutover).

> **Everything below is the superseded 3-lane fleet (2026-07-11/12/13)** — kept for the
> fp8-KV findings, sequential-start rationale, and GPU budget math, which still apply.

## [SUPERSEDED 2026-07-20] Daily setup (dual GPU) — 3-lane NVFP4 fleet (2026-07-11, fp8-KV added 2026-07-12)

All services run as systemd and auto-start on boot. **Daily driver = a 3-lane split**: fast (orchestrator/agentic), reasoning (thinking), coder — all NVFP4, all 256K ctx, all on vLLM 0.24.0 (`~/dev/vllm-024-test`). Superseded the 2× Ornith-replica setup (now [[project_blackwell_3lane_fleet]]).

| GPU | Service | Model | Port | util | Notes |
|-----|---------|-------|------|------|-------|
| 0 | `vllm-fast.service` | Ornith-1.0-35B-NVFP4 | :8040 | 0.33 | `fast`, tools (`qwen3_xml`), vision, orchestrator lane |
| 0 | `vllm-reasoning.service` | ThinkingCap-Qwen3.6-27B-NVFP4 | :8041 | 0.59 | `reasoning`, MTP k=1 (+51%), tools, thinking (util 0.62→0.59 on 2026-07-13 to fit embed-B on GPU0) |
| 0 | `embed-b.service` | Qwen3-Embedding-0.6B | :8004 | — | embed B (load-balanced w/ embed A) |
| 1 | `vllm-coder.service` | Qwen3-Coder-Next-NVFP4 | :8032 | 0.63 | `coder`, co-resident w/ Fish+embed |
| 1 | `embed-server.service` | Qwen3-Embedding-0.6B | :8001 | — | embed A |
| 1 | `protovoice-stack.service` | Fish S2-Pro TTS | :8092 | — | ~20GB, lazy-load |

**Embed is always-on, load-balanced across BOTH cards (2026-07-13):** `embed-server.service` (embed A, GPU1 :8001) + `embed-b.service` (embed B, GPU0 :8004), both `enabled` (survive reboot), same Qwen3-Embedding-0.6B. The 3-lane fleet had packed GPU0 solid (fast 0.33 + reasoning 0.62 → ~19 MiB free), which OOM'd embed-B; trimmed reasoning util **0.62→0.59** (frees ~3G, backup at `~/dev/.vllm-bump-review/unit-backups/vllm-reasoning.service.pre-embedb-util059-*`) so both embed lanes fit. Restart order if GPU0 lanes bounce: fast → reasoning (gated on fast `/health`) → embed-b. Reasoning reload was clean (~70s, fast untouched).

**All three lanes run `--kv-cache-dtype fp8` (2026-07-12).** fp8-KV is FIXED on 0.24.0/sm120 (overturns the old "broken on sm120" wall) — coherent, +2% speed, and **2–4× KV headroom**: fast 683K tok (2.61×), reasoning 872K (3.33×), coder 1.14M (4.33×). Depth-gate PASS: exact needle retrieval at 8K/32K/105K on all three, no rot. See [[project_vllm_0221_cuda13_migration]].

**Config invariants:** all lanes `--max-model-len 262144 --max-num-seqs 8 --enable-chunked-prefill --enable-prefix-caching --trust-remote-code` + the **sm120 recipe env** (`FLASHINFER_CUDA_ARCH_LIST=12.0f`, `CUDA_HOME=<cu13>`, cu13 in PATH, `NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, `VLLM_USE_TRITON_FP8_GEMM=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`). Marlin lanes use `--moe-backend marlin`; **MTP lanes drop the global `--moe-backend` and use `--linear-backend marlin`** so the MoE oracle picks cutlass (composes with the bf16 MTP head — global marlin was the blocker, not the head). Vision kept bf16 on the quant.

**⚠️ Sequential-start REQUIRED** — util is a fraction of the WHOLE card and the co-resident lanes race for memory on a shared GPU; the reasoning/coder units gate on the prior lane's `/health` via ExecStartPre. Don't parallel-restart same-GPU lanes.

**Gateway map** (homelab-iac#190, ava side): `protolabs/fast`→Ornith :8040, `protolabs/reasoning`→ThinkingCap :8041, `protolabs/coder`→Coder-Next :8032, old `reasoning`→`cloud` (DeepSeek V4). Eval judges target a local lane. WAN upload ~100 Mbps ([homelab-iac#176](https://github.com/protoLabsAI/homelab-iac/issues/176)) — publish big models per-file or cloud-quantize.

**Rollback to 2× Ornith replicas:** `vllm.service` (:8000) + `vllm-replica-b.service` (:8003) are on disk, **disabled** (relabeled ROLLBACK 2026-07-12); re-enable + start to revert. That setup ran both cards as `Ornith-1.0-35B-NVFP4` replicas (util 0.90/0.72), `protolabs/smart` round-robin, ~6500 tok/s aggregate — **replicas beat DP+EP/TP=2 on PCIe** (C32: DP+EP 3830 · single 3990 · 2 replicas ~6500); never shard a model that fits one card.

> **Everything below this line is HISTORICAL** — 2× Ornith replicas, dFlash, DiffusionGemma, AR-Gemma fast lane, 27B-MTP smart lane — all **superseded by the 3-lane fleet above**. Kept for breakdown material + the GPU/util/Fish-TTS budget math.

**2026-06-25: smart lane tried dFlash spec-decode, then REVERTED to MTP (see tradeoff note below).** For ~2.5h ran z-lab's block-diffusion **dFlash** draft (`z-lab/Qwen3.6-27B-DFlash`, 2B bf16, drafts for Qwen3.6-27B). Serves on **stock vLLM 0.22.1 — no bump, no PR, no `speculators` pip** (0.22.1 already ships `qwen3_dflash.py`/`DFlashQwen3ForCausalLM` + `v1/spec_decode/dflash.py`; the model card's "needs PR #40898 for interleaved SWA" note is stale for our build). Measured **74.6 → 106.9 decode tok/s (+43%)** single-stream in the live prod config (`-O3`, 225K, 512 seqs); tool-calling (`qwen3_xml`) + thinking verified clean, lossless (target verifies every accepted token). Only the `--speculative-config` line changed (MTP → `{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}`); `num_speculative_tokens=10` is the tuned peak (sweep 3/6/10/16 → 96.8/114.3/116.9/106.2 tok/s on a leaner GPU1 config). Backup unit at `~/dev/.vllm-bump-review/unit-backups/vllm.service.pre-dflash-*` — rollback to MTP = restore + `daemon-reload` + restart. Full work in `experiments/dflash/` (README/RESULTS/sweep.sh/run-dflash.sh/bench.sh/conc_bench.py/conc-driver.sh).

**⚠️ CONCURRENCY TRADEOFF → REVERTED to MTP same day (2026-06-25 21:57).** dFlash wins **single-stream only**. It does **not scale** — aggregate throughput plateaus ~540 tok/s regardless of load, while MTP scales near-linearly. Crossover ~C=3–4; by C=32 **MTP is ~3× faster**. Aggregate tok/s (dFlash N=10 vs MTP): C1 110.7/73.2 · C4 219.7/269.9 · C8 458.5/545.4 · C16 543.6/984.0 · C32 503.7/**1471.2**. Cause: dFlash drafts 10 tok/step @ ~16% accept (~11× verify-batch inflation, saturates compute under load); MTP drafts 1 @ ~79% (~2× inflation, stays efficient). `-O3` A/B = wash (543.6 vs 546.5 @ C16; earlier 116.9 was leaner ctx/seqs, not -O3). Eval quality held (custom 84% / FC 91% ≥ MTP baseline; lossless as expected).

**Why reverted:** initially kept dFlash on the assumption the smart lane is single-stream/interactive. Then **live Prometheus + LiteLLM-gateway traffic analysis** showed the opposite: when the lane is *actually busy* it's **parallel-agent fan-out at C=4–8 with a deep wait queue (18–24)** — real agents (protoAgent, Jon, QwenCode, langchainjs apps spawning parallel sub-agents), not single-stream. dFlash's single-stream win serves <5% of active samples; the C=4–8 regime that defines real heavy usage is exactly where MTP wins. So MTP is the correct default. Data source: Prometheus on ava (`100.101.189.45:9090`) — `vllm:num_requests_running{model="local"}` + `litellm_*` per-deployment/user-agent attribution. **dFlash stays a one-command swap-in for any genuinely single-stream lane** (draft cached at `models--z-lab--Qwen3.6-27B-DFlash`; set `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}'`). Full work + numbers in `experiments/dflash/` (README/RESULTS/sweep.sh/run-dflash.sh/bench.sh/conc_bench.py/conc-driver.sh).

Gemma fast lane has its own dFlash draft (`z-lab/gemma-4-26B-A4B-it-DFlash`) but needs unmerged vLLM **PR #41703** (gemma4 dflash not in 0.22.1) + MoE-spec is our historical −11% risk case.

**2026-06-14: fast lane REVERTED to AR Gemma 4 26B-A4B (DiffusionGemma demoted to on-demand) — now the shipped daily-driver fast lane.** A creative-writing head-to-head (`experiments/diffusion-creative-eval/`, 50 prompts vs human refs, MMD/Token-L2/slop + pairwise judge) found **AR Gemma 4 beats DiffusionGemma 64/36** on creative quality, lower MMD, less slop; DG's only edge was ~1.9× raw speed, which doesn't justify the quality loss + ops cost (DG can't guided-decode, no token streaming, OOM-on-restart, short-output collapse). Then two free wins on the AR lane: **(1)** official RedHat **FP8-Dynamic** quant (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`, per-tensor, pre-quantized — no `--dtype/--quantization`, loads 28 GB directly) = +4%; **(2)** **auto-tuned fused-MoE Triton config** for our exact shape (E=128,N=704, sm120; `models/moe-configs/` + `install-moe-configs.sh`) = +2%. Net **199 → 211 tok/s**, streams incrementally, guided-decode works. Unit: `/etc/systemd/system/vllm-fast.service` (serves the RedHat repo, util 0.72, NO `-O3` — regresses MoE ~25%, gemma4 parsers, `VLLM_USE_FLASHINFER_SAMPLER=0` + `TRITON_ATTN`). DG remains a one-command swap-in via `models/diffusion-fast-lane.sh` (+ backup unit in `~/dev/.vllm-bump-review/`). Also fixed an embed-server memory bloat found during the swap (6.4→1.7 GB; `experiments/rag-bench/serve_embed.py` MAX_BATCH + empty_cache + expandable_segments). **Everything below this line is the historical DiffusionGemma deployment, kept for the on-demand-DG runbook + the GPU1 util/Fish-TTS budget math.**

**2026-06-13: fast lane switched to DiffusionGemma 26B-A4B (text-diffusion), replacing Gemma 4.** Runs Google's day-zero diffusion runner via the `vllm/vllm-openai:gemma` **Docker container** (not in pip vLLM 0.22.1), launched by `models/diffusion-fast-lane.sh`: GPU1 `device=1`, :8002, `local-fast`, `--quantization fp8`, `--max-model-len 262144`, `--diffusion-config '{"canvas_length":256}'`, `--gpu-memory-utilization 0.68`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, gemma4 tool+reasoning parsers. FP8 (~26 GB) co-resides with Fish TTS + embed; **util 0.68 (not 0.72) + expandable_segments are required** — at 0.72 it OOMs during graph capture (the diffusion runner uses more transient memory than AR Gemma). Rollback: old AR unit at `~/dev/.vllm-bump-review/unit-backups/vllm-fast.service.pre-diffusion-*`. **Streaming caveat: it does NOT stream incrementally** — `stream:true` returns the whole canvas as one chunk (block-sized bursts beyond 256 tok), no token-by-token. It's a latency engine for long-form/low-concurrency content (~2–4× faster, prose quality on par with the Gemma-4 it's built on), NOT for short outputs / structured / guided / tools — full eval in `experiments/diffusiongemma/RESULTS.md`. The Gemma-4 notes below are historical, kept for the GPU1 util/Fish-TTS budget math (which still applies to the shared card).

**DiffusionGemma fast-lane runbook (gotchas hit + fixes, all baked into `models/diffusion-fast-lane.sh`):**
- **256-token truncation** — DG's `generation_config.json` defaults `max_new_tokens: 256`, which vLLM applies to any request that doesn't set `max_tokens` → outputs cut off mid-sentence. Fixed with **`--override-generation-config '{"max_new_tokens": 4096}'`** (merges in, keeps the diffusion params — do NOT use `--generation-config vllm`, that discards `max_denoising_steps`/`entropy_bound`/sampler config the runner needs).
- **Thinking leak / empty content** — DG's `chat_template.jinja` auto-enables thinking whenever there's a **system/developer message, tools, or `enable_thinking`** (an agent harness trips this). Thinking then either leaks `<|channel>thought…<channel|>` into `content` or eats the whole token budget (empty content). Fixed with a **no-think chat template** `models/diffusion-fast-nothink.jinja` (prepends `{%- set enable_thinking = false -%}`, forced via `--chat-template`) — thinking is now off regardless of client params. Plain user-only requests never tripped it; only system-message/agent traffic did.
- **Intermittent multi-second stalls** — first-hit Triton JIT compiles for shapes the startup graph-capture missed (`jit_monitor`). Fixed by `models/diffusion-fast-warmup.sh` wired as `ExecStartPost` (sweeps shapes on every start). Genuinely-novel shapes still pay a one-time compile.
- **Latency variance** — denoising steps per request vary wildly (observed 7 → 211); long/hard outputs cost multiples. `max_denoising_steps` (default 48/canvas, in `generation_config`) is the bounding lever — lower it via `--override-generation-config` to cap worst-case latency at a quality cost (not currently capped).
- **Residual `<|channel>` leak** still belongs to the gateway scrub (`_ThinkStripper`, homelab-iac#147) — route clients through `api.proto-labs.ai`/`ava:4000`, not direct to `:8002`, for belt-and-suspenders.

vllm-fast lives alone on GPU 1 as of 2026-05-03 — ComfyUI was disabled when image-gen work moved to avaLab. Gemma 4 26B MoE FP8 at `--gpu-memory-utilization 0.72 --max-model-len 262144` (256K). **util dropped 0.80 → 0.72 on 2026-05-30 because Fish TTS is now active** (S2-Pro server on `:8092`, ~19.8 GB on GPU 1) — the earlier "Fish TTS inactive" budget no longer holds. With Fish TTS (~19.8 GB) + qwen3-embed (~2 GB) co-resident, util 0.80 (76 GB) OOMs against ~73 GB free; 0.72 (68 GB) fits with headroom and still loads full 256K. KV math on this card: roughly `pool = (96 × util) − model_size − overhead`. If Fish TTS is ever stopped, util can go back to 0.80.

`--reasoning-parser gemma4` keeps Gemma's thinking off the `content` channel — verified clean across single-token-streaming reasoning/tool-call/constrained prompts (2026-05-30). **Caveat (2026-06-02): one client-side leak was reported via `protolabs/fast`** — reasoning landing on `content` as bare `thought…`/`<|channel>` text, the known upstream gemma4-parser class ([vllm#39885](https://github.com/vllm-project/vllm/issues/39885); related #38855/#39392), triggered by batched-delta streaming around the tool-result/multi-turn boundary. **Could not reproduce on the current build (0.20.1)** — clean across ~68 probes at default `--stream-interval 1` *and* under the documented trigger: `--stream-interval 20` + forced tool calls (`tool_choice: required`, `<|tool_call>` landing in a batched delta) + 16-way concurrent, raw-SSE inspected. The primary #39885 case (closed upstream) appears already handled here; the still-open residual PRs (#42875/#39898) target a narrower case we can't trigger, so a vLLM bump isn't load-bearing. Mitigation is a belt-and-suspenders gateway scrub ([homelab-iac#147](https://github.com/protoLabsAI/homelab-iac/issues/147), live + reversible via env flags) that strips Gemma's `<|channel>…<channel|>`/bare `thought` the way the Qwen `_ThinkStripper` handles `<think>` — assumed working, treated as observational since there's no synthetic repro to confirm against. Model + parser config on the box are correct and current — this is purely the upstream parser's streaming path.

**2026-05-31: LFM2.5-8B-A1B ("nano") evaluation — FAILED on Blackwell, Fish TTS retained.** Tried to drop in **LiquidAI/LFM2.5-8B-A1B** (hybrid conv+attn MoE, `Lfm2MoeForCausalLM`, 8.3B/1.5B-active, BF16 ~16.9 GB) on GPU 1 (port 8003) in place of the Fish/protovoice stack. Model downloaded fine and vLLM 0.20.1 registers `lfm2_moe`, but it **will not serve on sm120**: the unquantized MoE expert path routes to `flashinfer_cutlass_fused_moe`, which JIT-compiles an sm120 CUTLASS module and dies — `RuntimeError: No supported CUDA architectures found for major versions [12]` (flashinfer/compilation_context.py). Same FlashInfer-broken-on-Blackwell class as our attention findings, now in the fused-MoE path. **Not yet retried** with the flashinfer MoE backend disabled (candidate fix: force the Triton/native fused-MoE path instead of flashinfer-cutlass — investigate `VLLM_USE_FLASHINFER_MOE_*` / fused-moe backend selection). Weights kept in cache at `models--LiquidAI--LFM2.5-8B-A1B`. Fish/protovoice stack was stopped during the attempt and has been **re-enabled** (`protovoice-stack.service` active+enabled, :8092/:8093). Gemma stays at util 0.72.

Fast-lane history (preserved for context): originally Qwen 35B MoE FP8 official → heretic-FP8 (un-retired for uncensored prose, see [memory](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md)) → Gemma 4 26B MoE. Heretic was briefly reverted to and back on 2026-05-30 — it works (179 tok/s, clean with `--reasoning-parser qwen3`) but needs 32K output headroom for its trained-in think to close, so Gemma stays the default fast lane. The heretic HF card recommended a `logit_bias: {<think>:-100, </think>:-100}` clamp that corrupts generation — don't use, and ignore the HF README's workaround if you ever load that model. Gateway's `_ThinkStripper` (homelab-iac #26/#32/#35) handles `<think>...</think>` post-emit for any thinking-tagged model.

Tuned MoE kernels in `models/moe-configs/` symlink into vLLM's `fused_moe/configs/` via `bash models/install-moe-configs.sh` (run after fresh vLLM installs / upgrades).

## Running models

```bash
bash models/vllm-swap.sh qwen-35b           # speed king MoE FP8 official, 262K (180 tok/s)
bash models/vllm-swap.sh qwen-9b-fp8        # on-the-fly FP8 (140 tok/s)
bash models/vllm-swap.sh qwen-4b-fp8        # on-the-fly FP8 edge (140 tok/s)
bash models/vllm-swap.sh qwen-27b-int4      # daily driver, agentic (53 tok/s)
bash models/vllm-swap.sh qwen-27b-int4-mtp  # daily driver + MTP, chat/creative (70 tok/s)
bash models/vllm-swap.sh qwen-4b-int4       # edge deploy, fastest absolute (297 tok/s)
bash models/vllm-swap.sh qwen-4b            # LoRA base bf16 (155 tok/s)
bash models/vllm-swap.sh qwen-122b-fp8      # quality ceiling FP8 official TP=2 (112 tok/s)
bash models/vllm-swap.sh qwen-122b-int4     # quality ceiling INT4 TP=2 (122 tok/s)
bash models/vllm-swap.sh qwen-27b-fp8-tp2   # FP8 official TP=2 (70 tok/s, 131K)
```

## A/B chat (side-by-side model comparison)

```bash
python models/ab_chat.py                    # defaults: Daria gated :8043 vs base :8045
python models/ab_chat.py --a URL --am MODEL --al LABEL --b URL --bm MODEL --bl LABEL
python models/ab_chat.py --once "prompt"    # one-shot, scriptable
```

Single input → two columns, independent multi-turn histories, per-side latency. Works against any
two OpenAI-compatible endpoints (vLLM lanes, llama.cpp, custom servers). In-chat: `/sys <text>`
(system prompt both sides), `/clear`, `/save [path]` (JSON transcript → good sample material for
cards/blogs), `/quit`. Knobs: `--max-tokens --temp --timeout`. Zero deps beyond `requests`.

## Speed testing

```bash
bash models/speed-test.sh           # v1: 5 single-stream runs (800 tok gen) — legacy/continuity only
bash models/speed-test.sh 10        # 10 runs
bash models/speed-test.sh 3 short   # 3 short runs (200 tokens)

bash models/speed-test-v2.sh quick        # v2: 2 regimes × C{1,8}, ~10 min (release gate)
bash models/speed-test-v2.sh full         # v2: 4 ISL/OSL regimes × C{1,4,8,16,32}, ~60-90 min
bash models/speed-test-v2.sh depth        # v2: decode-at-depth ladder 4/16/32/64K × C{1,4} (~20 min;
                                          #     server needs max-model-len ≥ 64K; hybrid flat-curve + MTP-vs-depth data)
bash models/speed-test-v2.sh full 8003 x  # custom port + label

cd evals && bash run-ab-speed.sh qwen-4b-int4 5
```

v1 reports decode tok/s (1/TPOT), wall tok/s, TTFT, and TPOT from vLLM's `/metrics` endpoint — not wall-clock estimation. **v2 is the standard for published numbers** (InferenceMAX-style): client-side `vllm bench serve`, random dataset at fixed seed, regimes chat 1k/1k · context 8k/1k · gen 1k/8k · legacy 128/800, TTFT/TPOT p50/p99, aggregate tok/s, and goodput at TTFT≤2s+TPOT≤50ms. Single-stream-only numbers are banned from model cards (the dFlash lesson: single-stream wins can invert 3× under C=4–8 fan-out, which is what prod traffic actually is). JSONs land in `evals/results/speed-v2/` and feed the board's benchmark-result schema.

### Optimization flags (`-opt` configs)

Suffix any config with `-opt` to enable P1+P2:
- `--async-scheduling` — overlap scheduling with execution
- `--enable-prefix-caching` — reuse KV cache for repeated prefixes
- `--performance-mode interactivity` — auto-tune scheduler for latency
- `--kv-cache-dtype fp8` — halve KV cache memory, double context capacity

Measured impact (single-request, P1+P2 only): minimal (+1–3% tok/s). Real wins are under concurrent load and multi-turn (prefix caching). FP8 KV doubles context capacity.

### MTP speculative decoding (`-mtp` configs)

Native Qwen3.5 Multi-Token Prediction:

| Model | Baseline | + MTP | Gain | Tool calling |
|-------|:--------:|:-----:|:----:|:------------:|
| **27B FP8** | 50 tok/s | **73.6 tok/s** | **+47%** | TBD |
| **27B INT4** | 53 tok/s | **70 tok/s** | **+32%** | Works, T08 quality regresses |
| **9B** | 92 tok/s | **112 tok/s** | **+22%** | Works, no quality loss |
| **35B MoE** | 171 tok/s | 153 tok/s | -11% | N/A — slower |

MTP helps dense, hurts MoE (routing overhead > speculation savings). 9B + MTP safe for tool calling. 27B FP8 + MTP daily driver on GPU 0. 27B INT4 + MTP: chat/creative only, avoid for complex agentic. MoE FP8 env vars: `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency`.

### TP=2 tuning (122B, 35B-tp2)

NCCL env vars for PCIe (no NVLink): `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=8`.

- **`NCCL_P2P_DISABLE=1` enables stable CUDA graphs on TP=2 Blackwell PCIe**
- 122B INT4: 23 → **122 tok/s** (5.3×, stable over 10 runs)
- 35B MoE: 22 → **205 tok/s** (9.3×, stable over 10 runs)
- Root cause: ACS enabled on PCIe bridges corrupts P2P during CUDA graph replay
- Disabling P2P forces shared memory transport — slight overhead, fully stable
- TTFT: 3077 ms → 29 ms with prefix caching after warmup
- Power draw: 88–96 W per card — MoE is not power-bound. **⚠️ The "600 W limit" in this line
  was stale: both cards are capped at 375 W** by `nvidia-power-limit.service` (enabled,
  `nvidia-smi -pl 375`; `ExecStop` restores 600). Default/max is 600 W, min 150 W — so ~60%
  more power is available on request, deliberately not taken (energy budget).
- 35B TP=2: prefix caching fixed 1.8 s TTFT → 0.5 s (**−70%**), wall tok/s +25%
- `VLLM_USE_FLASHINFER_MOE_FP8` crashes on 122B FP8 (unsupported quant scheme) — don't use
- Previous finding that TP=2 needs enforce-eager was wrong — `NCCL_P2P_DISABLE=1` is the fix

## Running evals

**Standard model scorecard (use this for model decisions) — `evals/eval-model.sh`:**
```bash
cd evals
./eval-model.sh <label> <target-url> <model> [--quick|--full]   # → results/scorecard-<label>-<ts>/scorecard.{md,json}

./eval-model.sh ThinkingCap http://localhost:8041/v1 reasoning         # quick (default): the discriminating battery
./eval-model.sh Ornith-35B  http://localhost:8040/v1 fast --full       # + breadth suites
```
The **discriminating frontier battery** — the classic `run.sh profile` suites saturate (~0.95) on frontier
models, so this is the reusable standard: **quick** = claw(10, agentic) + reasoning_hard (solver-verified) +
LiveCodeBench (exec-graded) + function_call (schema); **full** adds instruction_following + structured_hard +
safety + creative_writing. Judge-free except claw, which uses an **independent cloud judge** (`protolabs/cloud`
= DeepSeek V4) so a local model never grades itself. Aggregator `runners/scorecard.py` also **flags any
llm_judge 0.5-fallbacks** (a dead judge can't masquerade as real scores). Knobs: `LCB_LIMIT=N`, `EVAL_THINKING=0`,
`JUDGE_MODEL`/`JUDGE_GATEWAY_URL`. **NOTE:** `evals/.env` pins the judge to a dead `:8000` (old replica) —
`eval-model.sh` overrides it to the cloud judge; if you use `run.sh --local` directly, repoint the judge first.

**Classic profiles / individual suites:**
```bash
./run.sh profile --name quick --model local    # ~15 min smoke test, 1 trial (NOTE: suites saturate on frontier models)
./run.sh profile --name full --model local     # comprehensive, full breadth, 1 trial

./run.sh claw --model local --tasks T02,T04,T06,T08 --port-offset 200
./run.sh custom --suite coding --model local --trials 1
./run.sh custom --suite reasoning_hard --model local --thinking   # the discriminating reasoning suite
./run.sh function-call --model local --all-suites
```

### Suites

| Suite | Tests | What it measures |
|-------|:-----:|-----------------|
| **claw-eval** | 52 EN | Agentic tool use (email, calendar, CRM, ops, finance) |
| **coding** | 10 | Generation (5) + analysis/review/security (5) |
| **instruction_following** | 5 | Constraint adherence, format compliance |
| **reasoning** | 5 | Math, logic puzzles, deduction, pattern recognition |
| **structured_output** | 5 | JSON, YAML, SQL, markdown tables, log parsing |
| **summarization** | 5 | Compression, action extraction, TL;DR |
| **safety** | 5 | Refusal, jailbreak resistance, PII, security review |
| **creative_writing** | 5 | Prose, narrative, character voice |
| **roleplay** | 5 | RPG GM quality, world building |
| **svg_generation** | 5 | SVG validity, accuracy, animation |
| **research** | 4 | Synthesis, conflicting sources, hallucination |
| **function_call** | 8 | Basic (5) + edge cases (3) |

Profiles: **quick** — 10 claw + custom + FC, 1 trial. **full** — 30 claw + all custom + FC, 1 trial (full breadth). **pass^3 dropped 2026-06-29** — breadth over 3× repetition; use `--trials N` only for targeted consistency checks.

## Model inventory (`/mnt/models`)

> ⚠️ The `tok/s` column below is **single-stream (C=1)** — internal reference only, **NOT a publishable number**
> (a C1 win can invert 3× under real C=4-8 fan-out — the dFlash lesson). For any card/blog/board number use
> `speed-test-v2.sh` (concurrency-swept aggregate + goodput + cache-warm + decode-at-depth-to-256K). See
> [[feedback_eval_prod_token_budget]] — the same honest-numbers bar applies to speed as to quality.

| Model | Size | tok/s (C1) | +MTP | Quick Score | Role |
|-------|------|:-----:|:----:|:-----------:|------|
| **Qwen 35B MoE FP8** | 35GB | **180** | — | — | Speed king, 262K ctx, single GPU (Qwen official) |
| **Qwen 9B FP8** | 19GB* | **141** | — | — | On-the-fly FP8 (+53% vs bf16) |
| **Qwen 4B FP8** | 8.8GB* | 141 | — | — | On-the-fly FP8 edge |
| **Qwen 27B FP8** | 29GB | 50 | **73.6** | — | Daily driver on GPU 0 (MTP on by default in vllm.service, 225K ctx; verified on hybrid-Mamba 3.6 / 0.22.1, 79% accept) |
| **Qwen 27B INT4** | 29GB | 53 | **70** | **86/103** | Alt daily driver |
| **Gemma 4 31B FP8** | 62GB | 42.9 | TBD | — | Dense alt; MTP config ready |
| **Qwen 122B FP8** | 119GB | **112** | — | — | Quality ceiling FP8 TP=2 |
| **Qwen 122B INT4** | 74GB | **122** | — | **89/103** | Quality ceiling INT4 TP=2 |
| **Qwen 27B FP8 TP=2** | 29GB | **70** | — | — | 131K ctx, TP=2 (Qwen official) |
| **Qwen 9B BF16** | 19GB | 92 | **112** | 72/103 | Fine-tune base (cold storage) |
| **Qwen 4B INT4** | 3.8GB | 297 | — | 56/103 | Edge deploy (fastest absolute) |

\* On-the-fly FP8 (`--quantization fp8`) loads bf16 from disk, quantizes during load.

Base models (0.8B, 2B, 4B) downloaded for pretraining. FP8 quants in `/mnt/models/quantized/` and on [HuggingFace protoLabsAI](https://huggingface.co/protoLabsAI).

Cold storage (`/mnt/data/models-cold/`): FLUX.2-klein 9B+base (100GB), Z-Image+Turbo (51GB), Voxtral-Mini-4B (17GB), OCR models (11.4GB). Image-gen models live here pending eventual reclaim — they belong on avaLab now.

## Blackwell constraints

**GPU thermals / power (characterized 2026-08-11 under sustained TP=2 load).** `GPU 0` =
PCI `01:00.0` = the **top card**, and it runs a consistent **~12 °C hotter** than GPU 1
(`03:00.0`) — 85/73 °C at peak, 76/64 °C at moderate load. Cause is airflow, not workload:
the lower card exhausts up into the top one. **This costs nothing today** — HW *and* SW
thermal slowdown both read **0 µs cumulative**, no throttle flags ever set, both cards hold
an identical 2805 MHz SM clock, and fans sit at 40–47%. Headroom is real, not marginal.
Power is capped at **375 W** (not the 600 W this doc used to claim) by
`nvidia-power-limit.service`; draw under load is ~260–380 W. Raising toward the 600 W
default is the obvious lever if a workload ever needs it — but it spends straight into the
top card's thermal delta, so measure GPU 0 first. Don't diagnose a hot top card as a fault.

**Chassis fans ARE measurable (2026-08-11) — `modprobe nct6775`, no reboot, no kernel args.**
It binds cleanly on this board (ASUS ROG Crosshair X670E Hero, BIOS 0922); ACPI does *not*
reserve the ports, so `acpi_enforce_resources=lax` is unnecessary. Chip = **NCT6799D at
0x2e:0x290** (`nct6799-isa-0290`). Persisted in `/etc/modules-load.d/nct6775.conf`.
node-exporter's hwmon collector picks it up with **zero config change** — 46 new
`node_hwmon_fan*` series flow straight to Prometheus on ava.

**And the finding that matters: the board's fan curve is GPU-blind.** Only 2 of 7 headers
are populated (fan2 ~1128 RPM, fan3 ~919 RPM) and both sit at **57% PWM** while GPU 0 runs
86 °C. Smart Fan IV (`pwm_enable=5`) regulates off the board's own thermistors — SYSTIN 52,
CPUTIN 54.5, PECI 57, all far under the 80 °C threshold — so the controller sees an idle-cool
machine and never ramps. The heat source is invisible to the thing moving the air. That is
*why* the top card runs hot, and it is a missing input, not a broken curve. Fixing it means
driving PWM from GPU temp — see the fan-curve task; **any such daemon must restore
`pwm_enable=5` on exit**, or a dead process leaves the fans pinned wherever it last wrote.
Ignore the `in0..in17` voltage ALARMs — lm-sensors ships no limits for this chip, so
min=max=0 flags everything.

**…and then MEASURED it, which killed the idea. Chassis airflow is NOT the constraint.**
Controlled A/B, load held constant at 10-14 concurrent requests, 4 min soak per arm:

```
                gpu0  gpu1   fan2 RPM  fan3 RPM
57% PWM (board)   85    72      1128       934
100% PWM          83    71      1662      1430
```

**+47% fan RPM bought ~1 °C** — and the 100% arm was carrying *more* load, so the real effect
is ≤1 °C. Don't build a GPU-driven chassis fan curve: it trades a genuine failure mode
(`pwm_enable=1` held by a process that can die) for nothing. The case already supplies more
air than the GPU coolers can use. It also means **the ~12 °C GPU0/GPU1 delta is card-to-card,
not case airflow** — GPU 1 exhausts into GPU 0's intake, and no amount of case fan can undo
that. The remaining lever is the cards' own fans, which idle at **52% / 44% while at 84/70 °C**
(read-only via `nvidia-smi`; `nvidia-settings` needs an X display this box doesn't have).

⚠️ **Thermal-A/B methodology, learned by getting it wrong:** the first run showed a glorious
86 → 47 °C "win" from ramping fans. It was entirely bogus — the lane went idle mid-test and
that was just residual heat bleeding off at zero load. Only caught because the sampler
printed `running=0.0`. **Always instrument load in a thermal experiment and hold it constant
across arms**; the confound is huge and points the same way as the hypothesis.

**2026-07-22: prod migrated to vLLM 0.25.0** (`~/dev/vllm-025`, clone of vllm-024-test + `pip install vllm==0.25.0`). Driven by poolside Laguna, whose card **requires 0.25.0+** — on 0.24.0 it garbled tool-calling + multi-turn agentic (band-aids: drop fp8-KV, patch `poolside_v1` regex #47311, sampling override — S/118B still borked). 0.25.0 fixes it NATIVELY (#42650 Blackwell attn + #47311 parser baked in; stock parser + fp8-KV clean, multi-turn stable). torch **stays 2.11.0+cu130**, flashinfer 0.6.12→0.6.13, sm120 recipe unchanged (first-load JIT ~4min). Behavior-preserving on Ornith: 206 tok/s (= 0.24.0), tools clean. `vllm-fast.service` repointed vllm-024-test→vllm-025 (rollback: `vllm-fast.service.pre-025-bak`; vllm-024-test env untouched). See [[reference_laguna_serving]].

**2026-07-11: prod migrated to vLLM 0.24.0** (both Ornith-35B-NVFP4 replicas, `vllm.service` + `vllm-replica-b.service`). torch **stays ==2.11.0+cu130** (0.24.0 pins it) — only vllm + flashinfer 0.6.11→0.6.12 + compressed-tensors 0.15→0.17 + `nvidia-cutlass-dsl[cu13]` moved. **Behavior-preserving:** same config (`--moe-backend marlin`, NO MTP, 256K, vision), MARLIN NVFP4 backend confirmed in both engine logs, FC parity 89% (vs ~91% baseline = noise). Env lives at `~/dev/vllm-024-test` (units repointed there: ExecStart + CUDA_HOME + PATH; sm120 recipe env unchanged). **Rollback = restore units from `~/dev/.vllm-bump-review/unit-backups/*.pre-0240-20260711-*` + daemon-reload + restart** (0.22.1 `~/dev/vllm-env` untouched). **Install debt:** `vllm-024-test` was a plain `pip install vllm==0.24.0` into a clone of prod `vllm-env`, NOT the hash-locked supply-chain review the 0.22.1 cutover got — harden before treating as canonical. Enables NVFP4+bf16-MTP composition (drop `--moe-backend`, oracle picks cutlass) — MTP left OFF pending a concurrency benchmark ([[project_qwen36_27b_smartlane_gate]]).

**2026-06-13: prod was vLLM 0.22.1 / CUDA 13** (was 0.20.1 / CUDA 12.8). torch 2.11+cu130, transformers 5.12. Supply-chain-reviewed hash-locked install. Two env vars are now REQUIRED on sm120 that 0.20.1 set automatically — without them you hit the FlashInfer sm75 crash:
- **`VLLM_USE_FLASHINFER_SAMPLER=0`** — 0.22.1 routes top-k logit sampling through FlashInfer's JIT, which rejects sm120. Now set in both units (`vllm.service`, `vllm-fast.service`) + `vllm-swap.sh`. Universal; needed for every model.
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN`** on the Gemma fast lane (belt-and-suspenders; 0.22.1 still auto-forces it for gemma4's heterogeneous head dims). Qwen auto-selects fine without it.
- Cutover was a venv directory-swap (`vllm-env` ⇄ `vllm-env-0.20.1-bak`) — this breaks pip console-script shebangs (`vllm`, `hf`); rewrite `vllm-env-OLD → vllm-env` across `bin/` after. `huggingface-cli` is gone in the new env → use `hf`. Rollback = swap dirs back + revert units; backups in `~/dev/.vllm-bump-review/unit-backups/`.

- CUDA graphs work on single GPU — don't use `--enforce-eager` (37–470% speedup).
- TP=2 needs `NCCL_P2P_DISABLE=1` (NOT `--enforce-eager` as previously thought). `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink).
- No xformers / Flash Attention — use PyTorch native SDPA.
- FlashInfer attention backend still auto-selects on some paths and crashes — Gemma4 auto-forces `TRITON_ATTN`; for others rely on auto-select + `VLLM_USE_FLASHINFER_SAMPLER=0`. Don't force `--attention-backend flashinfer`.
- **FlashInfer-on-sm120 is now CRACKED for the JIT/cutlass paths (2026-06-13, flashinfer 0.6.11 / CUDA 13).** The old "FlashInfer requires sm75 or higher" wall was a chain of arch-flag + nvcc-version + cccl-guard + OOM + linker issues, all fixable. The cutlass fused-MoE compiles+loads on sm120 with: `FLASHINFER_CUDA_ARCH_LIST=12.0f`, `CUDA_HOME=…/nvidia/cu13` (CUDA-13.3 nvcc, not system 12.8), `NVCC_APPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, `MAX_JOBS=4`, plus `cu13/lib64→lib` + bare `.so` symlinks. **Full recipe: `experiments/quantize/FLASHINFER-SM120-RECIPE.md`.** This is the door to FP8-KV cache (2× ctx), bf16 MoE, nano-LFM2, Mistral-128B — each still needs end-to-end serve validation.
- Unquantized **bf16 MoE** (e.g. 35B-A3B bf16) on 0.22.1 routes the fused-MoE to flashinfer-cutlass and hits the above wall — was fine on 0.20.1. Either apply the sm120 recipe, or use FP8/INT4 MoE (Triton path, works). `VLLM_USE_FLASHINFER_MOE_FP8=1` rejects Qwen's block-wise [128,128] FP8 quant scheme — don't use with Qwen3 FP8.
- `VLLM_USE_FLASHINFER_MOE_FP8=1` rejects Qwen's block-wise [128,128] FP8 quant scheme. Don't use with Qwen3 FP8 (122B, 35B, etc.).
- `-O3` (torch compile level 3) regresses MoE inference by ~25% — MoE routing is too dynamic for the compiler. Safe on dense (27B uses it), avoid on MoE.
- INT4 safe on dense, unstable on MoE (use BF16 for MoE).
- Capability cliff at 4B → 2B: sub-4B can't do agentic tool use.

## Thinking models — vLLM reasoning-parser gotcha

`--reasoning-parser qwen3` is greedy: any output before `</think>` is classified as `reasoning_content`. If the model fails to emit a closing `</think>` (common in long agent loops, amplified by `preserve_thinking=true`), the **entire answer** lands in `reasoning_content` and `content` is empty. Downstream consumers reading only `content` see a blank response. Upstream: [vllm-project/vllm#40528](https://github.com/vllm-project/vllm/issues/40528) — no fix yet.

Mitigations:
- **Gateway-side:** LiteLLM custom callback `thinking_normalizer.py` salvages `reasoning_content → content` when content is empty, strips inline `<think>...</think>` blocks, and exposes the raw trace as `reasoning` (OpenRouter convention). In `homelab-iac` `stacks/ai/config/litellm/callbacks/`.
- **Eval-side:** `claw-eval`'s `Message.text` accessor falls back to `reasoning_content` and rsplits on `</think>` for the primed-think case.

**vllm-fast must have `--reasoning-parser qwen3`** (2026-05-02 fix). Without it, heretic's trained-in always-thinks behavior emits `<think>` blocks straight into `content` for non-tool-call turns. The downstream cost: the protoCLI release-notes generator once published a leaked thinking trace as real release notes.

**Tool-call thinking observability** (2026-05-01): the `qwen3.5-tool-calling.jinja` template originally had "Fix 19" disabling `enable_thinking` when tools were present. Removing it lets the model emit `<think>thoughts</think><tool_call>...</tool_call>`; `qwen3_xml` parser extracts the tool call cleanly, gateway `_ThinkStripper` (primed-think branch) captures the thinking body to `metadata.thinking`. ~1 s extra per tool-call turn (200–400 tokens of thinking before the tool call). For latency-critical agent loops, route through `protolabs/fast` (the `-nothink` template).

## Secrets

All secrets in Infisical at `secrets.proto-labs.ai`. Never commit secrets. Gateway `start.sh` injects at runtime via Machine Identity.

## Storage

- `/mnt/models` — frequently-accessed model weights only (1TB NVMe, **257GB free / 71%** after
  the 2026-08-11 reclaim; was 30GB free / 97%)
- `/mnt/data` — datasets, checkpoints, outputs, cold model storage (2TB NVMe, **383GB free /
  78%** after the 2026-08-11 reclaim; was 88GB free / 95%)

**Reclaim 2026-08-11 — 523GB freed.** Removed, all verified zero-reference first:
superseded **poolside Laguna** trio (121G, lane replaced by DSV4 on 08-10) · **LTX-2 19B**
(293G — the generation before LTX-2.3's 22B; its 7 ComfyUI symlinks removed in the same
pass) · unused `ideogram-4-fp8` HF copy (26G — the *used* copy is `models-cold/Ideogram-4`) ·
`RedHatAI/gemma-4-31B-it-NVFP4` (22G) · HF `fishaudio/s2-pro` (11G — live copy is
`~/dev/fish-speech/checkpoints/s2-pro`) · `Ornith-1.0-9B-NVFP4` (11G) · gemma4 dspark/dflash/
eagle3 drafts (17G — DSV4's DSpark head is built in, no external draft) · canary-1b-flash +
non-turbo whisper-large-v3 (6G — protoVoice pins the **turbo** variant) · MiniCPM ×3 (6.8G,
vision retired here) · GLM-OCR (2.5G).

**Two rules that made this safe, worth reusing:**
- **`/mnt/models` and `/mnt/data` are separate filesystems**, so the `--local-dir` hardlink
  trap cannot span them. Verified `find -type f -links +1` returned **0** across the whole
  delete set before touching anything. Always run that check first.
- **Grep-absence is NOT proof of orphanhood.** `models-cold/ltx2-textenc` (13G) has *zero*
  grep-able references anywhere yet **is** the live text encoder for both LTX-2.3 workflows,
  reached only through one ComfyUI symlink. Enumerate symlink targets, don't just grep.
  **KEEP:** `ltx2-textenc`, `LTX-2.3`, `DeepSeek-V4-Flash-0731`, `gemma-3-12b`, `ACE-Step-1.5`.

**Pre-existing breakage found (NOT caused by the reclaim):** 16 dangling ComfyUI symlinks,
all predating this cleanup — targets `models--Comfy-Org--Qwen-Image_ComfyUI` (the documented
2026-05-03 hardlink incident), `models--Qwen--Qwen-Image-2512`, `models--Comfy-Org--ltx-2`,
`/mnt/models/anima`, plus stale `models--Lightricks--LTX-2.3` HF-cache links whose files are
gone (the working LTX-2.3 links point at `models-cold/LTX-2.3`). Image-gen workflows using
qwen-image or anima are already broken.
- `/mnt/data/models-cold/` — FLUX, Z-Image, Voxtral, OCR, **and the live DSV4-Flash
  checkpoint** the prod smart lane serves from (`DeepSeek-V4-Flash-0731`) — despite the
  "cold" name this path is load-bearing for prod; don't treat it as archive space
- `/` (OS drive) — **49% / 454GB free as of 2026-08-11** (was 99% / ~11GB; freed by Josh).
- **🚫 The Windows install is PERMANENT — do not propose reclaiming it (Josh, 2026-08-11).**
  `nvme1n1` (~931GB, EFI + MSR + NTFS + recovery) stays as-is. It is NOT free capacity, is
  not a reclaim candidate, and should not be offered as one when a drive fills up. Solve
  storage pressure by pruning models/data instead. The Windows DATA drive is **`/dev/sdb`**
  (1.8TB NTFS, label `Backup`) — never format or mount it either. (Node CLAUDE.md still
  calls it `/dev/sdd`; it enumerates as `sdb` now.)
- `/mnt/scratch` — logs, caches, docker volumes (disposable)
- `/mnt/pool` — **REMOVED 2026-05-28**: the 37TB mergerfs HDD pool (2x 20TB IronWolf Pro) was pulled and relocated to external housing on another machine. No bulk HDD storage on this node anymore; training corpora that lived here (incl. the 6.4T salm-duplex set) are now on the external box.

Full rules in [`/home/ava/dev/CLAUDE.md`](../CLAUDE.md) (the node-level CLAUDE.md). The hardlink trap from `huggingface-cli download --local-dir` is documented there — read it before cleaning HF caches.
