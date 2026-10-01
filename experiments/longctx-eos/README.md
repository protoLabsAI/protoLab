# longctx-eos — protoLab#36 repro harness

The smart lane sometimes ends a 60–80K-token reply with EOS partway through reasoning, leaving
`content` empty and `finish_reason=stop`. This directory reproduces that on a dedicated test lane.

- `lane.sh`: replica B's exact unit config (env + drop-ins) on GPU0 at `:8060`, so the gateway
  never routes to it. `SPEC_K=0|1|3 ./lane.sh`. Replica B must be stopped first.
- `trial.py`: paired trials (fixed seeds, so every arm gets the same prompts). Real vLLM source
  as the code text, 65–80K tokens, a nonce first so every prompt is cold, non-streaming,
  strict `json_schema`, `reasoning_effort=low`. `--no-schema` drops `response_format`.
- `driver.sh`: runs the arms back to back and restores replica B at the end. Unguarded;
  this is what overloaded replica A on 2026-10-01. Use `window.sh` instead.
- `window.sh`: guarded off-hours run. It aborts if A's waiting queue stays above 8 for 3 min or
  the 16:45Z deadline passes, and restores B on any exit. Arms: PR #44993 overlay at K=3, then
  `frequency_penalty 0.3` and `repetition_penalty 1.05`, 80 trials each. Traffic is quietest
  at 11Z–17Z (14–50 req/h vs 300–500 overnight).
- `pr44993-src.diff`: vllm PR #44993, the reasoning-boundary grammar fix in v0.27.0, for 0.25.1.
  Overlay, so prod's env is untouched:
  `cp -as $SP/vllm $OV/vllm`, swap the two patched files for real copies, `patch -p1`.
  Then `OVERLAY=$OV SPEC_K=3 ./lane.sh`.

## Results (2026-10-01, Swift-Qwen3.8-27B-NVFP4, vLLM 0.25.1, GPU0)

```
arm            n   early EOS     invalid JSON (non-blank replies)
k3 (prod)     40   5  (12.5%)    27/35
k3-noschema   40   5  (12.5%)    n/a (no schema requested)
k0 (no MTP)   38   2  (5.3%)      0/36
k1            not run (window closed)
```

- **MTP K=3 breaks strict `json_schema` output** (Fisher p = 8e-13). Malformed replies start
  `{"{"verdict"…` or `{{`. It looks like speculative tokens getting past the grammar at the
  reasoning-to-content switch. This is separate from the blank replies.
- **Early EOS happens without MTP too.** 5/40 vs 2/38 gives p = 0.43, which is underpowered.
  A ~2× effect at these rates needs ~150 per arm.
- **The schema doesn't cause the blanks.** Same 5/40 with and without it.
- This harness reproduces blanks at ~12%, about 3× the prod rate (4.1%), so it's the cheaper
  place to test.

`results/k3.discarded-v0.jsonl` comes from a first harness version whose prompt builder stopped
at the first oversized file (one prompt came out at 21K tokens). Excluded from the table.

## Upstream (researched 2026-10-01)

- Invalid JSON = vllm#34650/#48228. Under spec decode, `should_advance()` misses `</think>`, so
  the grammar never engages. The fix is PR #44993 (in v0.27.0).
- Blanks = vllm#55420. The model samples `<|im_end|>` inside `<think>`. It reproduces on
  llama.cpp, with MTP on or off, and on bf16 weights. The engine fix (PR #55562) was closed
  unmerged. Measured mitigations are in QwenLM/Qwen3.8#216. The gateway recovery is
  homelab-iac#290.

## Ops lesson

Running this with replica B out of the pool for ~3 h overloaded replica A (protolabs/smart
queue 47–61 deep, Vera's panels stuck). The first lane that fills up stays degraded: abandoned
gateway requests keep running (vLLM logged 0 aborts) and the gateway's in-memory least-busy
counts go stale after a lane restart. **Run long A/Bs off-hours, and watch the surviving
replica's queue during the window.**
