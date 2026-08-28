"""MTP draft-depth (K) sweep on real prompts, with acceptance read from /metrics.

Random-token load is banned here: a drafter cannot predict noise, so `--dataset-name
random` collapses acceptance and understates every speculative config (measured
independently by syv-ai/qwen38-27b-rtx3090 gotcha 6: the SAME server reads 35, 83 or
151 tok/s on random data depending on what the noise turns into). All prompts below are
coherent, and the long ones are drawn from the repo's own docs.

Reports per-stream decode tok/s, aggregate, TTFT p50, and MTP tokens-per-step, so a K
that wins on raw rate but loses acceptance is visible rather than hidden in the average.

  python evals/specdecode-residue/kbench.py <label> [--port N] [--conc 1,8]
"""
import argparse, json, re, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--port", type=int, default=8042)
ap.add_argument("--model", default="smart")
ap.add_argument("--conc", default="1,8")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--corpus", default="")
ap.add_argument("--long", action="store_true", help="prepend ~2.8k tokens of real document, the regime prod actually runs")
A = ap.parse_args()
BASE = f"http://127.0.0.1:{A.port}"

SHORT = [
    "Write a Python function that merges two sorted linked lists in place. Explain the "
    "invariant it maintains and why it is O(n+m) with no extra allocation.",
    "Explain how a modern SSD's flash translation layer works: mapping tables, garbage "
    "collection, wear levelling, write amplification and TRIM. Be precise.",
    "A systemd unit is `enabled` but the service will not start and journalctl shows "
    "nothing after the ExecStart line. List the checks you would run, in order, and say "
    "what each one rules out.",
    "Write a bash script that finds every file under a tree with more than one hard link, "
    "groups them by inode, and prints each group. Explain the failure modes.",
    "Describe the tradeoff between speculative decoding draft depth and acceptance rate. "
    "When does a deeper draft stop paying for itself?",
    "Refactor this into a pure function and explain what you changed and why:\n"
    "def total(items):\n    global TAX\n    s = 0\n    for i in items:\n        s += i.price\n"
    "    return s * (1 + TAX)",
    "Explain the difference between a CUDA graph capture and eager execution for LLM "
    "decode, and what makes a kernel unsafe to capture.",
    "Write pytest tests for a rate limiter that allows N requests per sliding window. "
    "Cover the boundary conditions explicitly.",
]


def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(x) for x in re.findall(
        rf"^{re.escape(n)}\{{[^}}]*\}} ([0-9.e+-]+)$", t, re.M))
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")


def stream(prompt, out):
    body = json.dumps({
        "model": A.model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": A.max_tokens, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time(); ttft = None; n = 0
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=1800)
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        delta = (d["choices"][0].get("delta") or {}).get("content")
        if delta:
            if ttft is None:
                ttft = time.time() - t0
            n += 1
    el = time.time() - t0
    out.append({"ttft": ttft or el, "tokens": n, "elapsed": el,
                "decode_tps": (n - 1) / (el - (ttft or 0)) if n > 1 and el > (ttft or 0) else 0})


if A.long:
    doc = open(A.corpus).read()[:11000]   # ~2.8k tokens of coherent prose
    SHORT = [doc + "\n\n" + q for q in SHORT]

print(f"# {A.label}  port={A.port}  max_tokens={A.max_tokens}")
print(f"{'C':>3} {'per-stream tok/s':>17} {'aggregate':>10} {'ttft p50':>9} {'tok/step':>9}")
results = {}
for c in [int(x) for x in A.conc.split(",")]:
    for warm in (True, False):          # first run after a restart reads 30-50% low (JIT)
        out = []
        d0 = metrics()
        ts = [threading.Thread(target=stream, args=(SHORT[i % len(SHORT)], out)) for i in range(c)]
        t0 = time.time()
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.time() - t0
        d1 = metrics()
    dn = d1[0] - d0[0]
    tok_step = ((d1[1] - d0[1]) / dn + 1) if dn else float("nan")
    per = statistics.median([o["decode_tps"] for o in out])
    agg = sum(o["tokens"] for o in out) / wall
    ttft = statistics.median([o["ttft"] for o in out])
    results[c] = {"per_stream": per, "aggregate": agg, "ttft_p50": ttft, "tok_per_step": tok_step}
    print(f"{c:>3} {per:>17.1f} {agg:>10.1f} {ttft:>8.3f}s {tok_step:>9.2f}", flush=True)

print(json.dumps({"label": A.label, "results": results}))
