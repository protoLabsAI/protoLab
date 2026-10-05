#!/usr/bin/env python3
"""Score a decision arm on deferral-bench-v0. Standalone: pandas + pyarrow + numpy, nothing else.

The point of this repo is that you do not need to run a model to evaluate a calibration or
deferral method. Load the shipped probability vectors, apply your method, score.

    python scorer.py                          # every shipped arm, every suite
    python scorer.py --arms smart-name Qwen3.5-4B-name
    python scorer.py --cascade Qwen3.5-4B-name --fallback smart-name

To score YOUR arm, write probs/<your-arm>/<suite>.test.parquet with columns
{id, y, probs} — `probs` a list of floats over that suite's options, in `meta/<suite>.json`
option order — and pass `--arms <your-arm>`. The `id` column must match the manifest.

Metrics, and why each is here:
  acc                accuracy. table stakes, and not what a router is bought for.
  ECE / Brier        is the probability meaningful as a NUMBER.
  AURC               is the probability meaningful as an ORDER — mean error over all coverages,
                     lower better. This is the one that decides deferral quality.
  sel@C              accuracy on the most-confident C% of items.
  cascade            accuracy when the arm answers its confident share and a fallback arm takes
                     the rest, simulated exactly (all arms ran the same items).

Reporting rule carried from the source experiment: never report ECE without AURC beside it. An
arm can reach ECE 0.006 by flattening to near-uniform at 10% accuracy — perfectly calibrated and
useless. Deferral quality is the check that catches it.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["injection", "sentiment", "routing", "langid"]


# ------------------------------------------------------------------ io
def load(arm, suite, split="test"):
    p = f"{HERE}/probs/{arm}/{suite}.{split}.parquet"
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    P = np.clip(np.stack(df["probs"].to_numpy()).astype(float), 1e-12, 1)
    P /= P.sum(1, keepdims=True)
    return {"P": P, "y": df["y"].to_numpy(), "id": df["id"].to_numpy(),
            "label_mass": df["label_mass"].to_numpy() if "label_mass" in df else None}


def arms():
    return sorted(os.path.basename(d) for d in glob.glob(f"{HERE}/probs/*") if os.path.isdir(d))


# ------------------------------------------------------------------ metrics
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def temperature(P, T):
    """Rescale a distribution by temperature. NOTE: for a 2-option decision this cannot change
    the ORDER of items by confidence, so AURC and sel@C are mathematically invariant to it."""
    return softmax(np.log(P) / T)


def fit_temperature(P, y):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    L = np.log(P)
    def nll(T):
        return -np.log(softmax(L / T)[np.arange(len(y)), y] + 1e-12).mean()
    return float(min(grid, key=nll))


def ece(P, y, bins=15):
    conf = P.max(1); corr = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(e)


def brier(P, y):
    return float(((P - np.eye(P.shape[1])[y]) ** 2).sum(1).mean())


def aurc(P, y):
    order = np.argsort(-P.max(1))
    err = (P.argmax(1)[order] != y[order]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def sel_acc(P, y, cov):
    k = max(1, int(round(cov * len(y))))
    order = np.argsort(-P.max(1))[:k]
    return float((P.argmax(1)[order] == y[order]).mean())


def cascade(Pa, ya, Pf, yf, defer):
    """Arm answers the top (1-defer) by its own confidence; the rest go to the fallback."""
    n = len(ya)
    k = n - int(round(defer * n))
    order = np.argsort(-Pa.max(1))
    keep, sent = order[:k], order[k:]
    hit = (Pa.argmax(1)[keep] == ya[keep]).sum() + (Pf.argmax(1)[sent] == yf[sent]).sum()
    return float(hit / n)


def boot_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(np.mean(v)), float(lo), float(hi)


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--suites", nargs="*", default=SUITES)
    ap.add_argument("--calibrate", action="store_true",
                    help="also fit a temperature on the calib split and report it in-domain")
    ap.add_argument("--cascade", default=None, help="arm to cascade FROM")
    ap.add_argument("--fallback", default="smart-name", help="arm to defer TO")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    A = a.arms or arms()
    rows = []

    hdr = (f"{'arm':20s} {'suite':10s} {'acc':>6s} {'ECE':>6s} {'Brier':>6s} "
           f"{'AURC':>7s} {'sel@90':>7s} {'sel@50':>7s}")
    if a.calibrate:
        hdr += f" | {'T-in':>5s} {'ECE T':>6s} {'AURC T':>7s}"
    print(hdr); print("-" * len(hdr))
    for arm in A:
        for s in a.suites:
            d = load(arm, s)
            if d is None:
                continue
            P, y = d["P"], d["y"]
            r = {"arm": arm, "suite": s, "n": len(y), "acc": float((P.argmax(1) == y).mean()),
                 "ece": ece(P, y), "brier": brier(P, y), "aurc": aurc(P, y),
                 "sel90": sel_acc(P, y, 0.9), "sel50": sel_acc(P, y, 0.5)}
            line = (f"{arm:20s} {s:10s} {r['acc']:6.3f} {r['ece']:6.3f} {r['brier']:6.3f} "
                    f"{r['aurc']:7.4f} {r['sel90']:7.3f} {r['sel50']:7.3f}")
            if a.calibrate:
                c = load(arm, s, "calib")
                if c is not None:
                    T = fit_temperature(c["P"], c["y"])
                    Pt = temperature(P, T)
                    r.update({"T_in": T, "ece_T": ece(Pt, y), "aurc_T": aurc(Pt, y)})
                    line += f" | {T:5.2f} {r['ece_T']:6.3f} {r['aurc_T']:7.4f}"
            rows.append(r)
            print(line)
        print()

    if a.cascade:
        fb = a.fallback
        print(f"== cascade: {a.cascade} answers its confident share, {fb} takes the rest ==")
        hdr = (f"{'suite':10s} {'alone':>6s} " +
               " ".join(f"{'d=' + str(d) + '%':>7s}" for d in [10, 20, 30, 50]) +
               f" {'fallback':>8s}")
        print(hdr); print("-" * len(hdr))
        for s in a.suites:
            da, df = load(a.cascade, s), load(fb, s)
            if da is None or df is None:
                continue
            assert (da["id"] == df["id"]).all(), "arms are not id-aligned on this suite"
            accs = [cascade(da["P"], da["y"], df["P"], df["y"], d / 100) for d in [10, 20, 30, 50]]
            print(f"{s:10s} {float((da['P'].argmax(1) == da['y']).mean()):6.3f} " +
                  " ".join(f"{v:7.3f}" for v in accs) +
                  f" {float((df['P'].argmax(1) == df['y']).mean()):8.3f}")
        print()

    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
