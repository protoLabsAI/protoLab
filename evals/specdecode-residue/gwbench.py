"""Same K comparison, but through the gateway on a real alias, at prod settings.

Standing rule: direct-to-lane probes do not reproduce caller behaviour. This drives
ava:4000 / protolabs/smart with thinking left ON (prod default) and a real token budget,
and records which deployment answered so a silent cloud fallback cannot masquerade as a
local win.

  python evals/specdecode-residue/gwbench.py <label> [--conc 1,8] [--long]
"""
import argparse, json, os, re, statistics, threading, time, urllib.request

# Background load is the confound in any gateway measurement on a live pair: the lanes
# carry real traffic, it drifts over minutes, and a sequential A/B silently attributes
# that drift to the arm measured later. Sample it and report it with every cell.
_LOAD = {"stop": False, "samples": []}


def _sample_load():
    while not _LOAD["stop"]:
        tot = 0.0
        for port in (8041, 8042):
            try:
                t = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5).read().decode()
                tot += sum(float(x) for x in re.findall(
                    r"^vllm:num_requests_running\{[^}]*\} ([0-9.e+-]+)$", t, re.M))
            except Exception:
                pass
        _LOAD["samples"].append(tot)
        time.sleep(2)

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--url", default="http://100.101.189.45:4000")
ap.add_argument("--model", default="protolabs/smart")
ap.add_argument("--conc", default="1,8")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--long", action="store_true")
ap.add_argument("--corpus", default="")
A = ap.parse_args()
KEY = os.environ.get("GATEWAY_API_KEY") or os.environ.get("LITELLM_API_KEY") or ""

PROMPTS = [
    "Write a Python function that merges two sorted linked lists in place. Explain the "
    "invariant it maintains and why it is O(n+m) with no extra allocation.",
    "Explain how a modern SSD's flash translation layer works: mapping tables, garbage "
    "collection, wear levelling, write amplification and TRIM. Be precise.",
    "A systemd unit is `enabled` but the service will not start and journalctl shows "
    "nothing after the ExecStart line. List the checks you would run, in order.",
    "Write a bash script that finds every file under a tree with more than one hard link, "
    "groups them by inode, and prints each group. Explain the failure modes.",
    "Describe the tradeoff between speculative decoding draft depth and acceptance rate. "
    "When does a deeper draft stop paying for itself?",
    "Explain the difference between a CUDA graph capture and eager execution for LLM "
    "decode, and what makes a kernel unsafe to capture.",
    "Write pytest tests for a rate limiter that allows N requests per sliding window. "
    "Cover the boundary conditions explicitly.",
    "Refactor this into a pure function and explain what changed:\n"
    "def total(items):\n    global TAX\n    s = 0\n    for i in items:\n        s += i.price\n"
    "    return s * (1 + TAX)",
]
if A.long:
    doc = open(A.corpus).read()[:11000]
    PROMPTS = [doc + "\n\n" + q for q in PROMPTS]


def stream(prompt, out):
    body = json.dumps({
        "model": A.model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": A.max_tokens, "temperature": 0, "stream": True,
    }).encode()
    t0 = time.time(); ttft = None; n = 0; served = None
    r = urllib.request.urlopen(urllib.request.Request(
        A.url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY}),
        timeout=1800)
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        served = served or d.get("model")
        ch = (d.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            if ttft is None:
                ttft = time.time() - t0
            n += 1
    el = time.time() - t0
    out.append({"ttft": ttft or el, "tokens": n, "elapsed": el, "served": served,
                "decode_tps": (n - 1) / (el - (ttft or 0)) if n > 1 and el > (ttft or 0) else 0})


print(f"# {A.label}  {A.model}  long={A.long}  max_tokens={A.max_tokens}")
print(f"{'C':>3} {'per-stream tok/s':>17} {'aggregate':>10} {'ttft p50':>9}  {'bg':>6}  served-by")
res = {}
for c in [int(x) for x in A.conc.split(",")]:
    for _warm in (True, False):
        out = []
        ts = [threading.Thread(target=stream, args=(PROMPTS[i % len(PROMPTS)], out)) for i in range(c)]
        _LOAD["samples"] = []; _LOAD["stop"] = False
        lt = threading.Thread(target=_sample_load, daemon=True); lt.start()
        t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.time() - t0
        _LOAD["stop"] = True; lt.join(timeout=3)
    per = statistics.median([o["decode_tps"] for o in out])
    agg = sum(o["tokens"] for o in out) / wall
    ttft = statistics.median([o["ttft"] for o in out])
    served = sorted({str(o["served"]) for o in out})
    # mean lane occupancy MINUS this benchmark's own c streams = background traffic
    occ = statistics.mean(_LOAD["samples"]) if _LOAD["samples"] else float("nan")
    bg = max(0.0, occ - c)
    res[c] = {"per_stream": per, "aggregate": agg, "ttft_p50": ttft, "served": served,
              "mean_occupancy": occ, "background": bg}
    print(f"{c:>3} {per:>17.1f} {agg:>10.1f} {ttft:>8.3f}s  bg={bg:>4.1f}  {','.join(served)}", flush=True)
print(json.dumps({"label": A.label, "results": res}))
