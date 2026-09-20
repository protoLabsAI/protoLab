#!/usr/bin/env python3
"""Paired bootstrap (2000 resamples of test items) for the calibration claims.
Temperatures are fitted once on calib splits and held fixed, so intervals cover test-sampling
noise only — not calib-sampling noise. That makes them slightly optimistic; stated, not hidden.
A delta is called only when its 95% interval excludes zero.
"""
import sys, numpy as np, metrics as M
rng = np.random.default_rng(20260919); B = 2000
arms = sys.argv[1:]
print(f"{'arm':20s} {'suite':10s} {'acc [95% CI]':>20s} | {'dECE T-in vs raw':>22s} | {'dECE T-loso vs raw':>22s}")
print("-" * 104)
tally = {"in_better": 0, "in_worse": 0, "loso_better": 0, "loso_worse": 0, "cells": 0}
for a in arms:
    data = {s: (M.load(a, s, "test"), M.load(a, s, "calib")) for s in M.SUITES}
    for s in M.SUITES:
        (L, y, _), (Lc, yc, _) = data[s]
        T_in = M.fit_T([(Lc, yc)])
        T_lo = M.fit_T([(data[o][1][0], data[o][1][1]) for o in M.SUITES if o != s])
        P = {t: M.softmax(L / T) for t, T in [("raw", 1.0), ("in", T_in), ("lo", T_lo)]}
        acc, din, dlo = [], [], []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y)); yy = y[i]
            e = {t: M.ece(p[i], yy) for t, p in P.items()}
            acc.append((P["raw"][i].argmax(1) == yy).mean()); din.append(e["in"] - e["raw"]); dlo.append(e["lo"] - e["raw"])
        ci = lambda v: np.percentile(v, [2.5, 97.5])
        def fmt(v):
            lo, hi = ci(v); m = np.mean(v)
            verdict = "better" if hi < 0 else "WORSE" if lo > 0 else "n.s."
            return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] {verdict:6s}", verdict
        a_lo, a_hi = ci(acc)
        fi, vi = fmt(din); fl, vl = fmt(dlo)
        tally["cells"] += 1
        tally["in_better"] += vi == "better"; tally["in_worse"] += vi == "WORSE"
        tally["loso_better"] += vl == "better"; tally["loso_worse"] += vl == "WORSE"
        print(f"{a:20s} {s:10s} {np.mean(acc):.3f} [{a_lo:.3f},{a_hi:.3f}] | {fi} | {fl}")
print()
print(f"cells={tally['cells']}  T-in: better {tally['in_better']}, WORSE {tally['in_worse']}  |  "
      f"T-loso: better {tally['loso_better']}, WORSE {tally['loso_worse']}  (rest not significant)")
