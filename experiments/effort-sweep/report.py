#!/usr/bin/env python3
"""Aggregate an effort sweep into the brevity-vs-quality table.

Per-item means are taken across trials FIRST, then averaged across items, so a
noisy item cannot dominate. Deltas vs the lane default (xhigh) are paired per item
and tested with an exact/Monte-Carlo sign-flip permutation test
(feedback_underpowered_is_not_null) — no t-test normality assumption.
"""
import json, sys, itertools, random
from pathlib import Path
from statistics import mean, median

ORDER = ["none", "low", "medium", "xhigh"]
BASE  = "xhigh"   # the lane's shipped default


def permutation_p(diffs, iters=200000, seed=20260908):
    """Two-sided sign-flip test on paired differences. Exact when 2^n <= iters."""
    d = [x for x in diffs if x is not None]
    n = len(d)
    if n == 0:
        return None
    obs = abs(sum(d))
    if n <= 18:
        cnt = sum(1 for signs in itertools.product((1, -1), repeat=n)
                  if abs(sum(s * v for s, v in zip(signs, d))) >= obs - 1e-12)
        return cnt / (2 ** n)
    rnd = random.Random(seed)
    cnt = sum(1 for _ in range(iters)
              if abs(sum(v if rnd.random() < .5 else -v for v in d)) >= obs - 1e-12)
    return cnt / iters


def load(path):
    return json.loads(Path(path).read_text())


def main(path):
    rows = load(path)
    _rank = {"reasoning_hard": 0, "lcb_medium": 1, "lcb_hard": 2}
    suites = sorted({r["suite"] for r in rows}, key=lambda x: (_rank.get(x, 9), x))
    efforts = [e for e in ORDER if any(r["effort"] == e for r in rows)]

    for suite in suites:
        sr = [r for r in rows if r["suite"] == suite]
        items = sorted({r["id"] for r in sr})
        # per-item mean score by effort
        by = {e: {i: [r["score"] for r in sr if r["effort"] == e and r["id"] == i]
                  for i in items} for e in efforts}

        print(f"\n=== {suite}  ({len(items)} items x {len(sr)//(len(items)*len(efforts))} trials"
              f" x {len(efforts)} efforts = {len(sr)} runs) ===\n")
        hdr = (f"{'effort':<9}{'score':>7}{'reason_tok':>11}{'answer_tok':>11}"
               f"{'compl_tok':>10}{'lat_s_p50':>10}{'thought%':>9}{'len_cap%':>9}{'err':>5}")
        print(hdr); print("-" * len(hdr))
        for e in efforts:
            er = [r for r in sr if r["effort"] == e]
            item_means = [mean(by[e][i]) for i in items if by[e][i]]
            ok = [r for r in er if not r.get("error")]
            rt = [r["reasoning_tokens"] for r in ok if (r["reasoning_tokens"] or 0) >= 0]
            at = [r["answer_tokens"] for r in ok if (r["answer_tokens"] or 0) >= 0]
            ct = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
            lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
            thought = [1 if (r["reasoning_tokens"] or 0) > 0 else 0 for r in ok]
            cap = [1 if r.get("finish_reason") == "length" else 0 for r in ok]
            print(f"{e:<9}{mean(item_means):>7.3f}{mean(rt):>11.0f}{mean(at):>11.0f}"
                  f"{mean(ct):>10.0f}{median(lat):>10.1f}"
                  f"{100*mean(thought):>8.0f}%{100*mean(cap):>8.0f}%"
                  f"{len(er)-len(ok):>5}")

        print(f"\npaired vs {BASE} (per-item, sign-flip permutation, two-sided):")
        print(f"{'effort':<9}{'d_score':>9}{'p':>9}{'d_reason_tok':>14}{'tok_saved%':>12}")
        base_items = by.get(BASE, {})
        b_rt = {i: mean([r["reasoning_tokens"] for r in sr
                         if r["effort"] == BASE and r["id"] == i
                         and (r["reasoning_tokens"] or 0) >= 0] or [0]) for i in items}
        for e in efforts:
            if e == BASE:
                continue
            diffs = [mean(by[e][i]) - mean(base_items[i])
                     for i in items if by[e][i] and base_items.get(i)]
            e_rt = {i: mean([r["reasoning_tokens"] for r in sr
                             if r["effort"] == e and r["id"] == i
                             and (r["reasoning_tokens"] or 0) >= 0] or [0]) for i in items}
            drt = mean([e_rt[i] - b_rt[i] for i in items])
            base_tot = mean([b_rt[i] for i in items]) or 1
            p = permutation_p(diffs)
            print(f"{e:<9}{mean(diffs):>+9.3f}{p:>9.4f}{drt:>+14.0f}{-100*drt/base_tot:>11.0f}%")

    print("\nlegend: score = judge-free (LCB exec partial-credit / reasoning_hard regex).")
    print("        thought% = share of runs that emitted ANY reasoning (adaptive thinking).")
    print("        len_cap% = share that hit the 32k budget without finishing (think-loop).")
    print("        latency is UNDER CONCURRENT LOAD — tokens are the primary brevity metric.")


if __name__ == "__main__":
    main(sys.argv[1])
