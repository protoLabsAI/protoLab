#!/usr/bin/env python3
"""Quantify the nonsense-refusal artifact, and attribute it to the clamp or to the base.

DARIA.md carried this as an open item before this probe ran: an "I can't browse the web..."
style assistant-register intrusion reported in both the HF and vLLM paths at low single-digit
rates on short prompts, unquantified. It matters to the clamp decision — if the clamp causes it, that is a cost on
the other side of the ledger from the slop reduction.

The Round 4 / powered-A/B corpora cannot answer it: those are long-form EQ-Bench prompts and
the artifact is reported on SHORT ones. Checking there returned 0/96 per arm, which is a
statement about the prompt distribution, not about the artifact. So this probes short prompts
specifically, against the same two lanes that differ only in DARIA_ENABLE.

  python refusal_probe.py --reps 12 --on http://127.0.0.1:8045/v1 --off http://127.0.0.1:8046/v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = "/mnt/data/abliterate/creative-vectors/refusal_probe.json"
_lock = threading.Lock()

# Short, open creative asks — the shape the artifact was observed on. Deliberately varied in
# register (some imperative, some question-like) because a question-shaped prompt is the more
# plausible trigger for an assistant-register intrusion.
PROMPTS = [
    "Write a haiku about a locked door.",
    "Describe the smell of a hardware store.",
    "One paragraph: a man misses his train on purpose.",
    "What does the sea sound like from inside a house?",
    "Write the last line of a novel that has no first line.",
    "A short scene: two strangers share an umbrella.",
    "Describe November in six sentences.",
    "Tell me about a chair nobody sits in.",
    "Write an opening sentence that makes a reader uneasy.",
    "Give me a paragraph about the inside of a piano.",
    "Who lives at the end of the road?",
    "Write about the moment before a phone rings.",
]

REFUSAL = re.compile(
    r"\b(I (?:can(?:no|')t|am unable to|don't have the ability to) (?:browse|access|search)"
    r"|as an AI\b|I'm an AI\b|I am an AI\b|as a language model"
    r"|I don't have (?:real-time|access to the internet)"
    r"|I cannot fulfill|I can't assist with|I'm sorry, but I|I don't have personal)",
    re.I,
)


def call(url, prompt, max_tokens, timeout=300):
    body = dict(model="daria", messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.7, min_p=0.1)
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def wilson(k, n, z=1.96):
    """Wilson score interval — correct at the low rates and small k this artifact lives at,
    where a normal approximation would produce a negative lower bound."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--on", default="http://127.0.0.1:8045/v1")
    ap.add_argument("--off", default="http://127.0.0.1:8046/v1")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    lanes = {"clamp_on": a.on, "clamp_off": a.off}
    # arm innermost — see clamp_ab.py: an arm-major list time-separates the arms.
    jobs = [(arm, rep, i, p) for rep in range(a.reps)
            for i, p in enumerate(PROMPTS) for arm in lanes]
    print(f"{len(PROMPTS)} short prompts x {a.reps} reps x 2 arms = {len(jobs)} calls", flush=True)

    recs, failed, t0 = [], [], time.time()

    def work(job):
        arm, rep, i, prompt = job
        try:
            out = call(lanes[arm], prompt, a.max_tokens)
        except Exception as e:
            print(f"  FAIL {arm} rep{rep} p{i}: {e!r}", flush=True)
            with _lock:
                failed.append((arm, rep, i))
            return
        with _lock:
            recs.append(dict(arm=arm, rep=rep, prompt_id=i, prompt=prompt, text=out,
                             refusal=bool(REFUSAL.search(out))))

    with ThreadPoolExecutor(max_workers=a.concurrency * 2) as ex:
        list(ex.map(work, jobs))

    if failed:
        import collections as _c
        sys.exit(f"ABORT: {len(failed)} of {len(jobs)} calls failed "
                 f"({dict(_c.Counter(a_ for a_, _, _ in failed))}) — a Wilson interval over an "
                 f"incomplete arm understates the rate. Re-run.")
    json.dump(recs, open(a.out, "w"))
    print(f"wrote {a.out} ({len(recs)} calls) in {(time.time()-t0)/60:.1f} min\n")

    for arm in sorted(lanes):
        rs = [r for r in recs if r["arm"] == arm]
        k = sum(r["refusal"] for r in rs)
        lo, hi = wilson(k, len(rs))
        print(f"{arm:>10s}  {k}/{len(rs)} = {k/max(len(rs),1)*100:5.2f}%  "
              f"(95% CI {lo*100:.2f}-{hi*100:.2f}%)")

    hits = [r for r in recs if r["refusal"]]
    if hits:
        print(f"\nexamples ({len(hits)} total):")
        for r in hits[:5]:
            m = REFUSAL.search(r["text"])
            print(f"  [{r['arm']}] {r['prompt'][:44]!r} -> ...{r['text'][max(0,m.start()-40):m.end()+60]!r}")
    else:
        print("\nno refusal-register intrusions detected in either arm")


if __name__ == "__main__":
    main()
