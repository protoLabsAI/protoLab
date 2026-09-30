#!/usr/bin/env python3
"""Powered A/B: does the v6 clamp reduce slop, on top of repetition_penalty 1.15?

Round 4 compared these two arms at n=3 and reported a -19% slop reduction as the clamp's one
surviving contribution. Re-analysing that data per-run (`clamp_decision.py`) shows the slop
index has a run-to-run sd of ~0.81, so a 1.12 delta sits at 1.4 sd with overlapping arm
ranges — directionally consistent, but not distinguishable from noise. At that effect size
(d ~ 1.4) 80% power needs about n=9 per arm.

So: 9 runs per arm, same 32 EQ-Bench prompts, two concurrently-served vLLM lanes that differ
in exactly one environment variable.

**No judge.** The open question is the slop index, which is a deterministic function of the
text. Judging would add the 16k-budget coverage problem and a cloud bill for a number that
cannot move the decision — Round 4 already settled the rubric axis at -0.16 (inside noise).

Serving both arms concurrently is deliberate: it puts them under the same GPU contention and
the same wall clock, so nothing systematic separates them but the clamp.

  python clamp_ab.py --reps 9 --on http://127.0.0.1:8045/v1 --off http://127.0.0.1:8046/v1
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BENCH = "/mnt/scratch/downloads/creative-writing-bench"
OUT = "/mnt/data/abliterate/creative-vectors/clamp_ab_pieces.json"

_lock = threading.Lock()


def prompts():
    """The 32 prompt/seed-modifier pairs exactly as the recorded runs used them."""
    runs = json.load(open(f"{BENCH}/creative_bench_runs.json"))["base_final__base"]
    out = []
    for it in runs["creative_tasks"].values():
        for t in it.values():
            mods = list((t.get("results_by_modifier") or {}).keys())
            if mods:
                out.append((t["prompt_id"], mods[0], t["base_prompt"].replace("<SEED>", mods[0])))
    return out


def call(url, prompt, timeout=1800):
    body = dict(model="daria", messages=[{"role": "user", "content": prompt}],
                max_tokens=4096, temperature=0.7, min_p=0.1)
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--on", default="http://127.0.0.1:8045/v1", help="clamp ON lane")
    ap.add_argument("--off", default="http://127.0.0.1:8046/v1", help="clamp OFF lane")
    ap.add_argument("--concurrency", type=int, default=8, help="per lane; matches --max-num-seqs")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    P = prompts()
    lanes = {"clamp_on": a.on, "clamp_off": a.off}
    # ARM IS THE INNERMOST LOOP, deliberately. ThreadPoolExecutor consumes FIFO, so an
    # arm-major list drains one whole arm before the other starts and the arms end up
    # time-separated — exactly the confound concurrent serving exists to remove. The first
    # run of this study did that: pieces 0-276 of 576 were all clamp_on (caught in review).
    jobs = [(arm, rep, pid, mod, text)
            for rep in range(a.reps) for pid, mod, text in P for arm in lanes]
    print(f"{len(P)} prompts x {a.reps} reps x {len(lanes)} arms = {len(jobs)} pieces", flush=True)

    recs, done, failed, t0 = [], [0], [], time.time()

    def work(job):
        arm, rep, pid, mod, text = job
        try:
            out = call(lanes[arm], text)
        except Exception as e:
            print(f"  FAIL {arm} rep{rep} p{pid}: {e!r}", flush=True)
            with _lock:
                failed.append((arm, rep, pid))
            return
        with _lock:
            recs.append(dict(config=f"{arm}#{rep}", arm=arm, run=rep,
                             premise=f"{pid}|{mod[:24]}", prompt=text, text=out))
            done[0] += 1
            if done[0] % 32 == 0:
                el = (time.time() - t0) / 60
                print(f"  {done[0]}/{len(jobs)} in {el:.1f} min "
                      f"(eta {el / done[0] * (len(jobs) - done[0]):.1f} min)", flush=True)
                json.dump(recs, open(a.out + ".tmp", "w"))

    # Both lanes are driven at once so they share GPU contention and wall clock.
    with ThreadPoolExecutor(max_workers=a.concurrency * len(lanes)) as ex:
        list(ex.map(work, jobs))

    if os.path.exists(a.out + ".tmp"):
        os.remove(a.out + ".tmp")

    # A lane-specific failure biases every downstream number, so refuse to bank a partial
    # corpus rather than publish an A/B that silently compared unequal arms.
    if failed:
        by_arm = collections.Counter(arm for arm, _, _ in failed)
        sys.exit(f"ABORT: {len(failed)} of {len(jobs)} calls failed ({dict(by_arm)}) — "
                 f"corpus NOT written. Re-run; a lane-specific failure would bias the result.")

    json.dump(recs, open(a.out, "w"))
    print(f"wrote {a.out} ({len(recs)} pieces) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
