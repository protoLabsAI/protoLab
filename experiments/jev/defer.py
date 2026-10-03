#!/usr/bin/env python3
"""Deferral quality — the metric the whole question turns on. numpy only.

A decision model earns its place by knowing when to hand off. Accuracy cannot measure that.
Three views, all computed from the SAME per-item probability files metrics.py scores:

  1. risk-coverage      AURC + selective accuracy at 95/90/80/50% coverage, per arm x suite
  2. calibration-vs-deferral   does temperature scaling actually improve the RANKING, or only
                               the numbers? T changes the argmax never, but it DOES reorder
                               max-prob across items, so AURC can move. Paired bootstrap.
  3. cascade            tiny arm answers when confident, defers the rest to the 27B lane.
                        Accuracy vs. % deferred, and the deferral rate needed to match the
                        27B alone. Test items are id-aligned across every arm (verified), so
                        this is an exact simulation, not an estimate.

Usage:  python defer.py            # all arms, writes results/deferral.{txt,json}
"""
import json, glob, os, sys
import numpy as np
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = M.SUITES
REF = "smart-name"          # the fallback a cascade defers TO
COVS = [0.95, 0.90, 0.80, 0.50]
DEFER_GRID = [0.0, 0.10, 0.20, 0.30, 0.50]
B = 2000
rng = np.random.default_rng(20260920)


def arms():
    a = sorted(os.path.basename(d) for d in glob.glob(f"{HERE}/results/*") if os.path.isdir(d))
    return [x for x in a if M.load(x, SUITES[0], "test") is not None]


def temps(arm, suite):
    """(T_in, T_loso) fitted exactly as metrics.py fits them — calib splits only."""
    ca = M.load(arm, suite, "calib")
    T_in = M.fit_T([(ca[0], ca[1])])
    others = [(M.load(arm, o, "calib")[0], M.load(arm, o, "calib")[1]) for o in SUITES if o != suite]
    return T_in, M.fit_T(others)


def rc_curve(P, y):
    """Selective accuracy at every coverage, items ordered by confidence descending."""
    order = np.argsort(-P.max(1))
    corr = (P.argmax(1)[order] == y[order]).astype(float)
    return np.cumsum(corr) / np.arange(1, len(corr) + 1)


def sel_at(P, y, cov):
    c = rc_curve(P, y)
    return float(c[max(0, int(round(cov * len(y))) - 1)])


def cascade(Pa, ya, Pr, yr, defer):
    """Arm answers the top (1-defer) by its own confidence; the rest go to the reference arm."""
    n = len(ya)
    k = n - int(round(defer * n))
    order = np.argsort(-Pa.max(1))
    keep, sent = order[:k], order[k:]
    hit = (Pa.argmax(1)[keep] == ya[keep]).sum() + (Pr.argmax(1)[sent] == yr[sent]).sum()
    return float(hit / n)


def main():
    A = arms()
    out = {"risk_coverage": [], "temp_vs_defer": [], "cascade": []}
    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)

    # ---------- 1. risk-coverage ----------
    emit("== 1. RISK-COVERAGE (raw probabilities, no calibration) ==")
    emit("selective accuracy = accuracy on the items the arm is most confident about.")
    emit("AURC = mean error over all coverages; LOWER is better. Bootstrap CI = 2000 resamples.")
    emit()
    hdr = f"{'arm':20s} {'suite':10s} {'acc':>6s} {'sel@95':>7s} {'sel@90':>7s} {'sel@80':>7s} {'sel@50':>7s} {'AURC':>6s} {'[95% CI]':>16s}"
    emit(hdr); emit("-" * len(hdr))
    for a in A:
        for s in SUITES:
            L, y, _ = M.load(a, s, "test")
            P = M.softmax(L)
            boot = []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y))
                boot.append(M.aurc(P[i], y[i]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            r = {"arm": a, "suite": s, "acc": float((P.argmax(1) == y).mean()),
                 "aurc": M.aurc(P, y), "aurc_ci": [float(lo), float(hi)],
                 **{f"sel{int(c*100)}": sel_at(P, y, c) for c in COVS}}
            out["risk_coverage"].append(r)
            emit(f"{a:20s} {s:10s} {r['acc']:6.3f} {r['sel95']:7.3f} {r['sel90']:7.3f} "
                 f"{r['sel80']:7.3f} {r['sel50']:7.3f} {r['aurc']:6.3f} [{lo:6.3f},{hi:6.3f}]")
        emit()

    # ---------- 2. does calibration improve deferral? ----------
    emit("== 2. DOES TEMPERATURE SCALING IMPROVE DEFERRAL? (dAURC vs raw; negative = better) ==")
    emit("Temperature never moves the argmax, so accuracy is fixed. It CAN reorder confidence")
    emit("across items, so AURC can move. If it does not, calibration is cosmetic for routing.")
    emit()
    hdr = f"{'arm':20s} {'suite':10s} {'AURC raw':>9s} | {'dAURC T-in':>22s} | {'dAURC T-loso':>22s}"
    emit(hdr); emit("-" * len(hdr))
    tally = {"in_better": 0, "in_worse": 0, "lo_better": 0, "lo_worse": 0, "cells": 0}
    for a in A:
        for s in SUITES:
            L, y, _ = M.load(a, s, "test")
            T_in, T_lo = temps(a, s)
            P = {t: M.softmax(L / T) for t, T in [("raw", 1.0), ("in", T_in), ("lo", T_lo)]}
            d_in, d_lo = [], []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y)); yy = y[i]
                base = M.aurc(P["raw"][i], yy)
                d_in.append(M.aurc(P["in"][i], yy) - base)
                d_lo.append(M.aurc(P["lo"][i], yy) - base)
            def fmt(v):
                lo, hi = np.percentile(v, [2.5, 97.5]); m = float(np.mean(v))
                verdict = "better" if hi < 0 else "WORSE" if lo > 0 else "n.s."
                return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] {verdict:6s}", verdict, m, float(lo), float(hi)
            fi, vi, mi, li, hi_ = fmt(d_in); fl, vl, ml, ll, hl = fmt(d_lo)
            tally["cells"] += 1
            tally["in_better"] += vi == "better"; tally["in_worse"] += vi == "WORSE"
            tally["lo_better"] += vl == "better"; tally["lo_worse"] += vl == "WORSE"
            out["temp_vs_defer"].append({"arm": a, "suite": s, "aurc_raw": M.aurc(P["raw"], y),
                                         "T_in": T_in, "T_loso": T_lo,
                                         "d_in": mi, "d_in_ci": [li, hi_], "d_in_verdict": vi,
                                         "d_loso": ml, "d_loso_ci": [ll, hl], "d_loso_verdict": vl})
            emit(f"{a:20s} {s:10s} {M.aurc(P['raw'], y):9.4f} | {fi} | {fl}")
    emit()
    emit(f"cells={tally['cells']}  T-in: better {tally['in_better']}, WORSE {tally['in_worse']}  |  "
         f"T-loso: better {tally['lo_better']}, WORSE {tally['lo_worse']}  (rest not significant)")
    emit()

    # ---------- 3. cascade ----------
    emit(f"== 3. CASCADE: arm answers what it is confident about, defers the rest to {REF} ==")
    emit("Exact simulation: every arm ran the same id-aligned test items (verified).")
    emit("'defer%' is the share of items that cost a 27B call. 'match@' = smallest deferral")
    emit(f"rate on this grid whose accuracy reaches {REF} alone (-- = never on the grid).")
    emit()
    hdr = (f"{'arm':20s} {'suite':10s} {'alone':>6s} " +
           " ".join(f"{'d=' + str(int(d*100)) + '%':>7s}" for d in DEFER_GRID) +
           f" {'27B alone':>9s} {'match@':>7s}")
    emit(hdr); emit("-" * len(hdr))
    Lr = {s: M.load(REF, s, "test") for s in SUITES}
    for a in A:
        if a == REF: continue
        for s in SUITES:
            L, y, _ = M.load(a, s, "test")
            Pa = M.softmax(L)
            Lrr, yr, _ = Lr[s]; Pr = M.softmax(Lrr)
            ref_acc = float((Pr.argmax(1) == yr).mean())
            accs = [cascade(Pa, y, Pr, yr, d) for d in DEFER_GRID]
            match = next((d for d, v in zip(DEFER_GRID, accs) if v >= ref_acc), None)
            out["cascade"].append({"arm": a, "suite": s, "ref": REF, "ref_acc": ref_acc,
                                   "defer_grid": DEFER_GRID, "acc": accs,
                                   "match_at": match})
            emit(f"{a:20s} {s:10s} {accs[0]:6.3f} " +
                 " ".join(f"{v:7.3f}" for v in accs) +
                 f" {ref_acc:9.3f} " + (f"{int(match*100):6d}%" if match is not None else "     --"))
        emit()

    # ---------- 4. pooled cascade vs the 27B alone, paired bootstrap ----------
    emit(f"== 4. POOLED CASCADE vs {REF} ALONE (all four suites, equal n) ==")
    emit("Macro-average over 4 suites of 300 items. Paired bootstrap resamples items WITHIN")
    emit("each suite, so the comparison is on the same items both arms saw. A positive delta")
    emit(f"means the cascade beat {REF} while paying only 'defer%' of the 27B calls.")
    emit()
    hdr = f"{'arm':20s} {'defer%':>6s} {'pooled acc':>10s} {'d vs 27B alone':>24s}"
    emit(hdr); emit("-" * len(hdr))
    ref_pool = float(np.mean([(M.softmax(Lr[s][0]).argmax(1) == Lr[s][1]).mean() for s in SUITES]))
    emit(f"{REF + ' alone':20s} {'100':>6s} {ref_pool:10.3f} {'--':>24s}")
    for a in A:
        if a == REF: continue
        for d in DEFER_GRID:
            deltas = []
            per = []
            for s in SUITES:
                L, y, _ = M.load(a, s, "test"); Pa = M.softmax(L)
                Lrr, yr, _ = Lr[s]; Pr = M.softmax(Lrr)
                per.append((Pa, y, Pr, yr))
            acc = float(np.mean([cascade(*p, d) for p in per]))
            for _ in range(B):
                ds = []
                for Pa, y, Pr, yr in per:
                    i = rng.integers(0, len(y), len(y))
                    ds.append(cascade(Pa[i], y[i], Pr[i], yr[i], d) - (Pr[i].argmax(1) == yr[i]).mean())
                deltas.append(np.mean(ds))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            verdict = "BEATS 27B" if lo > 0 else "worse" if hi < 0 else "n.s."
            out["cascade"].append({"arm": a, "pooled": True, "defer": d, "acc": acc,
                                   "delta": float(np.mean(deltas)), "ci": [float(lo), float(hi)],
                                   "verdict": verdict})
            emit(f"{a:20s} {int(d*100):5d}% {acc:10.3f} "
                 f"{np.mean(deltas):+.3f} [{lo:+.3f},{hi:+.3f}] {verdict:9s}")
        emit()

    # ---------- 5. threshold transfer: the deployable version of section 3/4 ----------
    emit("== 5. THRESHOLD TRANSFER (the deployable cascade) ==")
    emit("Sections 3-4 defer a FIXED FRACTION of test items — that silently uses the test set to")
    emit("locate the cut. In production you fit one confidence threshold offline and live with")
    emit("whatever coverage it yields. Here tau is the target-quantile of max-prob on the arm's")
    emit("CALIB split, applied unchanged to test. 'got%' is the deferral rate it actually bought.")
    emit()
    hdr = (f"{'arm':20s} {'suite':10s} {'want%':>5s} {'got%':>5s} {'acc(tau)':>8s} "
           f"{'acc(oracle)':>11s} {'gap':>7s}")
    emit(hdr); emit("-" * len(hdr))
    for a in A:
        if a == REF: continue
        for s in SUITES:
            Lc, yc, _ = M.load(a, s, "calib"); Pc = M.softmax(Lc)
            L, y, _ = M.load(a, s, "test"); Pa = M.softmax(L)
            Lrr, yr, _ = Lr[s]; Pr = M.softmax(Lrr)
            for d in [0.10, 0.20, 0.30]:
                tau = float(np.quantile(Pc.max(1), d))
                sent = Pa.max(1) < tau
                got = float(sent.mean())
                hit = (Pa.argmax(1)[~sent] == y[~sent]).sum() + (Pr.argmax(1)[sent] == yr[sent]).sum()
                acc_t = float(hit / len(y))
                acc_o = cascade(Pa, y, Pr, yr, d)
                out.setdefault("threshold_transfer", []).append(
                    {"arm": a, "suite": s, "want": d, "tau": tau, "got": got,
                     "acc_tau": acc_t, "acc_oracle": acc_o})
                emit(f"{a:20s} {s:10s} {int(d*100):4d}% {100*got:4.0f}% {acc_t:8.3f} "
                     f"{acc_o:11.3f} {acc_t - acc_o:+7.3f}")
        emit()

    json.dump(out, open(f"{HERE}/results/deferral.json", "w"), indent=1)
    open(f"{HERE}/results/deferral.txt", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote results/deferral.json and results/deferral.txt")


if __name__ == "__main__":
    main()
