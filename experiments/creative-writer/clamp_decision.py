#!/usr/bin/env python3
"""Does the v6 clamp earn its keep on top of repetition_penalty 1.15?

Round 4 settled the rubric axis: base+rep -> daria is -0.16 points against a pooled run-to-run
sd of 1.44, i.e. nothing. What it did NOT do is put an error bar on the one axis where the
clamp does move the needle — the deterministic slop index — or quantify the costs. A single
slop number per arm cannot answer "keep it or drop it"; the delta has to clear the same kind
of noise band the rubric delta failed to clear.

This re-analyses the ALREADY-BANKED Round 4 pieces. No generation, no judge, no GPU: the slop
index and the n-gram metrics are deterministic functions of the text, so the answer is already
sitting in the files.

    arm A  base            rep 1.0, no clamp      repeat_base_clean_scored.json
    arm B  base+rep1.15    rep 1.15, no clamp     repeat_scored.json      base#*
    arm C  tau0+rep1.15    rep 1.15 + v6 clamp    repeat_scored.json      tau0+rep1.15#*

(arm B lives under the label "base#*" because that run inherited the lane's repetition_penalty
default — the contamination Round 4 caught. Its rubric means, 11.64/11.68/12.31, match the
published base+rep1.15 row 58.2/58.4/61.5, while the clean file's 10.52/10.63/10.64 match the
true base row 52.6/53.2/53.2. The mapping is checked at load, not assumed.)

Usage:  python clamp_decision.py
"""
from __future__ import annotations

import collections
import json
import re
import statistics as st
import sys

VEC = "/mnt/data/abliterate/creative-vectors"
BENCH = "/mnt/scratch/downloads/creative-writing-bench"

# Same construction compare_runs.py uses, so slop numbers stay comparable with RESULTS.md.
_SP = json.load(open(f"{BENCH}/data/slop_phrase_prob_adjustments.json"))
_W = [1.0 - adj for _, adj in _SP]
_MX = max(_W)
SLOP = {w.lower(): s / _MX for (w, _), s in zip(_SP[:600], _W[:600])}

# Single owner: refusal_probe.py. Two copies had already drifted apart (it matched one extra
# alternative), which would have made "0/288 here" and "0/144 there" different measurements
# reported as the same one.
from refusal_probe import REFUSAL  # noqa: E402

DEGEN_4GRAM = 10  # RESULTS.md defines the gate as a 4-gram repeated >=10x in one piece


def slop_1kw(t: str) -> float:
    tl = t.lower()
    return sum(s * len(re.findall(r"\b" + re.escape(w) + r"\b", tl))
               for w, s in SLOP.items()) / max(len(t.split()), 1) * 1000


def worst_4gram(t: str) -> int:
    w = [x.strip('.,;:!?"\'').lower() for x in t.split() if x.strip()]
    g = collections.Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return max(g.values()) if g else 0


def load_arms():
    clean = json.load(open(f"{VEC}/repeat_base_clean_scored.json"))
    mixed = json.load(open(f"{VEC}/repeat_scored.json"))
    arms = {
        "base":         [r for r in clean if r["config"].startswith("base#")],
        "base+rep1.15": [r for r in mixed if r["config"].startswith("base#")],
        "tau0+rep1.15": [r for r in mixed if r["config"].startswith("tau0+rep1.15#")],
    }
    # Guard the file->arm mapping against the published table rather than trusting labels.
    means = {k: st.mean(r["rubric"] for r in v) for k, v in arms.items()}
    if not (means["base"] < means["base+rep1.15"] and means["base"] < means["tau0+rep1.15"]):
        sys.exit(f"arm mapping looks wrong (rubric means {means}) — check the source files")
    return arms


def _validate(arms):
    """Refuse to compute on a corpus that cannot support the comparison.

    clamp_ab.py now aborts rather than banking a partial run, but older files predate that
    and a missing arm would otherwise surface as a KeyError halfway through a statistic —
    or worse, as unequal arms silently compared.
    """
    missing = {"clamp_on", "clamp_off"} - set(arms)
    if missing:
        sys.exit(f"corpus is missing arm(s) {sorted(missing)} — cannot run the comparison")
    counts = {k: collections.Counter(r["config"] for r in v) for k, v in arms.items()}
    sizes = {k: sorted(set(c.values())) for k, c in counts.items()}
    if any(len(v) != 1 for v in sizes.values()):
        sys.exit(f"unequal pieces per run: {sizes} — arms are not comparable")
    if len({len(c) for c in counts.values()}) != 1:
        sys.exit(f"unequal run counts per arm: "
                 f"{ {k: len(c) for k, c in counts.items()} } — arms are not comparable")


def per_run(records):
    runs = collections.defaultdict(list)
    for r in records:
        runs[r["config"]].append(r)
    out = []
    for cfg in sorted(runs):
        pieces = runs[cfg]
        texts = [p["text"] for p in pieces]
        w4 = [worst_4gram(t) for t in texts]
        out.append(dict(
            cfg=cfg,
            n=len(texts),
            slop=st.median(slop_1kw(t) for t in texts),
            # unjudged corpora (the powered A/B) carry no rubric — it is not needed there
            rubric=(st.mean(p["rubric"] for p in pieces)
                    if all("rubric" in p for p in pieces) else float("nan")),
            words=st.mean(len(t.split()) for t in texts),
            degen=sum(1 for x in w4 if x >= DEGEN_4GRAM),
            worst4=max(w4),
            refusals=sum(1 for t in texts if REFUSAL.search(t)),
        ))
    return out


def permutation_p(a, b, iters=None):
    """Exact two-sided permutation test on run-level means.

    With 9 runs per arm there are C(18,9)=48620 splits, so the null distribution can be
    enumerated rather than sampled — no normality assumption, no scipy, and no ambiguity
    about whether n=9 is 'enough' for a t-test.
    """
    from itertools import combinations
    pool = list(a) + list(b)
    n = len(a)
    obs = abs(st.mean(a) - st.mean(b))
    splits = list(combinations(range(len(pool)), n))
    hits = 0
    for idx in splits:
        left = [pool[i] for i in idx]
        right = [pool[i] for i in range(len(pool)) if i not in set(idx)]
        if abs(st.mean(left) - st.mean(right)) >= obs - 1e-12:
            hits += 1
    return obs, hits / len(splits), len(splits)


def cmd_ab(path):
    """Analyse the powered two-lane A/B (clamp_ab.py output)."""
    recs = json.load(open(path))
    arms = collections.defaultdict(list)
    for r in recs:
        arms[r["arm"]].append(r)
    _validate(arms)
    print(f"powered A/B: {len(recs)} pieces, arms {{{', '.join(f'{k}:{len(v)}' for k, v in arms.items())}}}\n")

    print(f"{'arm':>10s} {'run':>4s} {'slop':>6s} {'words':>6s} {'degen':>6s} {'worst4':>7s} {'refuse':>7s}")
    print("-" * 50)
    summary = {}
    for name in sorted(arms):
        rows = per_run(arms[name])
        for i, r in enumerate(rows):
            print(f"{name if i == 0 else '':>10s} {i:>4d} {r['slop']:6.2f} {r['words']:6.0f} "
                  f"{r['degen']:6d} {r['worst4']:7d} {r['refusals']:7d}")
        summary[name] = rows

    print()
    for name, rows in summary.items():
        s_ = [r["slop"] for r in rows]
        n = sum(r["n"] for r in rows)
        print(f"{name:>10s}  slop {st.mean(s_):5.2f} +/- {st.stdev(s_):.2f}  "
              f"words {st.mean(r['words'] for r in rows):4.0f}  "
              f"degen {sum(r['degen'] for r in rows)}/{n}  "
              f"refuse {sum(r['refusals'] for r in rows)}/{n}")

    on = [r["slop"] for r in summary["clamp_on"]]
    off = [r["slop"] for r in summary["clamp_off"]]
    obs, p, nsplits = permutation_p(on, off)
    delta = st.mean(on) - st.mean(off)
    print(f"\nCLAMP CONTRIBUTION (clamp_off -> clamp_on), slop index, n={len(on)} runs/arm:")
    print(f"  delta        {delta:+.2f} per 1k words ({delta / st.mean(off) * 100:+.1f}%)")
    print(f"  permutation  p = {p:.4f}  (exact, {nsplits} splits)")
    print(f"  ranges       on {min(on):.2f}-{max(on):.2f}   off {min(off):.2f}-{max(off):.2f}"
          f"   {'(disjoint)' if max(on) < min(off) or max(off) < min(on) else '(overlap)'}")
    print(f"  verdict      {'REAL at p<0.05' if p < 0.05 else 'NOT distinguishable from noise'}")


def cmd_holdout(path):
    """Is the de-slop effect real, or an artifact of tuning on the metric we report?

    The tau/lam operating point was selected in phase2_gated_probe.py using slop_1kw over the
    TOP 600 phrases — the same index clamp_decision reports. That is circular: the powered
    p-value establishes the effect REPRODUCES across fresh generations, not that it exists on
    anything the clamp was not fitted to.

    The phrase list has 50084 entries. Bands past 600 were never used in tuning or in any
    published number, so they are a genuine held-out test of the same underlying claim —
    "the clamp reduces slop-register language" — rather than "the clamp reduces these 600
    strings". Same banked pieces, no generation, no judge.
    """
    bands = [(0, 600, "tuned-on (top 600)"), (600, 1200, "held-out 600-1200"),
             (1200, 5000, "held-out 1200-5000"), (5000, 20000, "held-out 5000-20000")]
    recs = json.load(open(path))
    arms = collections.defaultdict(list)
    for r in recs:
        arms[r["arm"]].append(r)
    _validate(arms)

    print(f"held-out validation on {len(recs)} banked pieces (no generation, no judge)\n")
    print(f"{'band':>22s} {'phrases':>8s} {'off':>7s} {'on':>7s} {'delta':>8s} {'pct':>7s} {'perm p':>8s}")
    print("-" * 70)
    for lo, hi, label in bands:
        sub = dict(zip((w.lower() for w, _ in _SP[lo:hi]),
                       (s_ / _MX for s_ in _W[lo:hi])))

        def idx(t, table=sub):
            tl = t.lower()
            return sum(v * len(re.findall(r"\b" + re.escape(w) + r"\b", tl))
                       for w, v in table.items()) / max(len(t.split()), 1) * 1000

        per_arm = {}
        for name, rs in arms.items():
            runs = collections.defaultdict(list)
            for r in rs:
                runs[r["config"]].append(r["text"])
            per_arm[name] = [st.median(idx(t) for t in v) for v in runs.values()]
        on, off = per_arm["clamp_on"], per_arm["clamp_off"]
        obs, pv, _ = permutation_p(on, off)
        d = st.mean(on) - st.mean(off)
        pct = d / st.mean(off) * 100 if st.mean(off) else float("nan")
        star = "  <-- tuned here" if lo == 0 else ("  *" if pv < 0.05 else "")
        print(f"{label:>22s} {hi-lo:>8d} {st.mean(off):7.2f} {st.mean(on):7.2f} "
              f"{d:+8.2f} {pct:+6.1f}% {pv:8.4f}{star}")
    print("\n* = holds at p<0.05 on phrases the operating point was never fitted to.")


def main():
    if "--holdout" in sys.argv:
        i = sys.argv.index("--holdout")
        return cmd_holdout(sys.argv[i + 1] if len(sys.argv) > i + 1
                           else f"{VEC}/clamp_ab_pieces.json")
    if "--ab" in sys.argv:
        i = sys.argv.index("--ab")
        return cmd_ab(sys.argv[i + 1] if len(sys.argv) > i + 1
                      else f"{VEC}/clamp_ab_pieces.json")
    arms = load_arms()
    print(f"{'arm':>14s} {'run':>4s} {'slop':>6s} {'rubric':>7s} {'words':>6s} "
          f"{'degen':>6s} {'worst4':>7s} {'refuse':>7s}")
    print("-" * 64)
    summary = {}
    for name, recs in arms.items():
        rows = per_run(recs)
        for i, r in enumerate(rows):
            print(f"{name if i == 0 else '':>14s} {i:>4d} {r['slop']:6.2f} {r['rubric']:7.2f} "
                  f"{r['words']:6.0f} {r['degen']:6d} {r['worst4']:7d} {r['refusals']:7d}")
        summary[name] = rows

    print(f"\n{'arm':>14s} {'slop mean':>10s} {'sd':>6s} {'rubric':>7s} {'sd':>6s} "
          f"{'degen':>6s} {'refuse':>7s}")
    print("-" * 60)
    for name, rows in summary.items():
        s = [r["slop"] for r in rows]
        b = [r["rubric"] for r in rows]
        print(f"{name:>14s} {st.mean(s):10.2f} {st.stdev(s):6.2f} {st.mean(b):7.2f} "
              f"{st.stdev(b):6.2f} {sum(r['degen'] for r in rows):3d}/{sum(r['n'] for r in rows):<3d}"
              f" {sum(r['refusals'] for r in rows):3d}/{sum(r['n'] for r in rows):<3d}")

    # The decision: clamp ON vs OFF, both with the sampler.
    b = [r["slop"] for r in summary["base+rep1.15"]]
    c = [r["slop"] for r in summary["tau0+rep1.15"]]
    pooled_sd = ((st.stdev(b) ** 2 + st.stdev(c) ** 2) / 2) ** 0.5
    delta = st.mean(c) - st.mean(b)
    print(f"\nCLAMP CONTRIBUTION (base+rep1.15 -> tau0+rep1.15), slop index:")
    print(f"  delta      {delta:+.2f} per 1k words  ({delta / st.mean(b) * 100:+.1f}%)")
    print(f"  pooled sd  {pooled_sd:.2f}   (n=3 per arm)")
    print(f"  |delta|/sd {abs(delta) / pooled_sd:.1f}"
          f"   -> {'CLEARS the noise band' if abs(delta) > 2 * pooled_sd else 'INSIDE noise'}")
    print(f"  arm ranges  B {min(b):.2f}-{max(b):.2f}   C {min(c):.2f}-{max(c):.2f}"
          f"   {'(disjoint)' if max(c) < min(b) or max(b) < min(c) else '(OVERLAP)'}")


if __name__ == "__main__":
    main()
