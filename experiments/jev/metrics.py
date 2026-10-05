#!/usr/bin/env python3
"""Score decision runs: accuracy, calibration, deferral. numpy only.

Three calibration conditions per arm x suite:
  raw     probabilities as the model emitted them
  T-in    temperature fitted on THIS suite's calib split          (needs labelled in-domain data)
  T-loso  temperature fitted on the OTHER suites' calib splits    (zero-shot-calibration proxy:
                                                                    does a fix learned elsewhere transfer?)
Temperature scaling never changes the argmax, so accuracy is identical across conditions; it
moves NLL/Brier/ECE. AURC measures whether confidence RANKS items well — the deferral property.
"""
import json, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["injection", "sentiment", "routing", "langid"]

def load(arm, suite, split):
    p = f"{HERE}/results/{arm}/{suite}.{split}.jsonl"
    if not os.path.exists(p): return None
    rs = [json.loads(l) for l in open(p)]
    P = np.clip(np.array([r["probs"] for r in rs], float), 1e-12, 1)
    return np.log(P), np.array([r["y"] for r in rs]), np.mean([r["label_mass"] for r in rs])

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def nll(L, y, T=1.0):
    return -np.log(softmax(L / T)[np.arange(len(y)), y] + 1e-12).mean()

def fit_T(pairs):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    best = min(grid, key=lambda T: sum(nll(L, y, T) * len(y) for L, y in pairs))
    return float(best)

def ece(P, y, bins=15):
    conf = P.max(1); corr = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return e

def aurc(P, y):
    order = np.argsort(-P.max(1)); err = (P.argmax(1)[order] != y[order]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))

def sel_acc(P, y, cov=0.8):
    k = max(1, int(round(cov * len(y)))); order = np.argsort(-P.max(1))[:k]
    return float((P.argmax(1)[order] == y[order]).mean())

def macro_f1(pred, y, k):
    f = []
    for c in range(k):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn == 0: continue
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def score(arm):
    data = {s: (load(arm, s, "test"), load(arm, s, "calib")) for s in SUITES}
    rows = []
    for s in SUITES:
        te, ca = data[s]
        if te is None or ca is None: continue
        L, y, mass = te; k = L.shape[1]
        T_in = fit_T([(ca[0], ca[1])])
        others = [(data[o][1][0], data[o][1][1]) for o in SUITES if o != s and data[o][1] is not None]
        T_lo = fit_T(others) if others else float("nan")
        r = {"arm": arm, "suite": s, "n": len(y), "k": k, "label_mass": mass,
             "acc": float((L.argmax(1) == y).mean()), "f1": macro_f1(L.argmax(1), y, k),
             "T_in": T_in, "T_loso": T_lo}
        for tag, T in [("raw", 1.0), ("in", T_in), ("loso", T_lo)]:
            P = softmax(L / T)
            r[f"nll_{tag}"] = nll(L, y, T); r[f"ece_{tag}"] = ece(P, y)
            r[f"brier_{tag}"] = float(((P - np.eye(k)[y]) ** 2).sum(1).mean())
        P = softmax(L)
        r["aurc"] = aurc(P, y); r["sel80"] = sel_acc(P, y, 0.8)
        rows.append(r)
    return rows

if __name__ == "__main__":
    arms = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(f"{HERE}/results/*") if os.path.isdir(d))
    allr = [r for a in arms for r in score(a)]
    json.dump(allr, open(f"{HERE}/results/summary.json", "w"), indent=1)
    hdr = f"{'arm':20s} {'suite':10s} {'k':>3s} {'acc':>6s} {'f1':>6s} {'sel@80':>6s} {'aurc':>6s} | {'ece raw':>7s} {'ece T-in':>8s} {'ece T-loso':>10s} | {'T-in':>5s} {'T-loso':>6s} {'mass':>5s}"
    print(hdr); print("-" * len(hdr))
    for r in allr:
        print(f"{r['arm']:20s} {r['suite']:10s} {r['k']:>3d} {r['acc']:6.3f} {r['f1']:6.3f} {r['sel80']:6.3f} {r['aurc']:6.3f} | "
              f"{r['ece_raw']:7.3f} {r['ece_in']:8.3f} {r['ece_loso']:10.3f} | {r['T_in']:5.2f} {r['T_loso']:6.2f} {r['label_mass']:5.3f}")
