#!/usr/bin/env python3
"""Rung 0 — is the hard half SCALE or RANKING? numpy only, no new inference.

Finding 7 split a probability's job in two: the ORDERING of items (discrimination, fixed at
decode time) and the SCALE that maps confidence to a number you threshold. Post-hoc calibration
can only touch the second. This script asks how much is even on the table.

  1. oracle ceiling   isotonic fitted ON THE TEST SET (an oracle, deliberately) is the best ANY
                      monotone recalibration could do. raw - oracle = the entire scale budget.
                      Whatever is left is ranking, and only training moves it.
  2. Murphy split     top-label Brier = reliability - resolution + uncertainty. Reliability is
                      the scale error; resolution is the discrimination the arm actually has.
  3. conditional T    log T = w . phi(x) from DECODE-TIME features only (entropy, margin,
                      label_mass, log k, prompt length), fitted leave-one-suite-out on calib and
                      applied to a suite it never saw. The honest test of "is the right map
                      predictable from what the model can already see".
                      A per-ITEM temperature can reorder items, so unlike a global T it moves
                      AURC too -- it gets a shot at both halves.
  4. free ensemble    for arms run in both readout formats, does format agreement rank better
                      than confidence alone? Two passes we already paid for.

Usage:  python calib_transfer.py           # writes results/transfer.{txt,json}
"""
import json, glob, os
import numpy as np
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = M.SUITES
B = 2000
BINS = 15
rng = np.random.default_rng(20260920)
LOG_T_LO, LOG_T_HI = np.log(0.05), np.log(20.0)

FEATS = ["entropy", "margin", "label_mass", "log_k", "log_len"]


# ---------------------------------------------------------------- loading
def text_len(suite, split):
    """Prompt length by item id — a decode-time feature, no label involved."""
    out = {}
    for line in open(f"{HERE}/data/{suite}.{split}.jsonl"):
        r = json.loads(line)
        out[r["id"]] = len(r["text"])
    return out


def load_full(arm, suite, split):
    p = f"{HERE}/results/{arm}/{suite}.{split}.jsonl"
    if not os.path.exists(p):
        return None
    rs = [json.loads(l) for l in open(p)]
    P = np.clip(np.array([r["probs"] for r in rs], float), 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    L = np.log(P)
    y = np.array([r["y"] for r in rs])
    mass = np.array([r.get("label_mass", 1.0) for r in rs], float)
    lens = text_len(suite, split)
    ln = np.array([lens[r["id"]] for r in rs], float)
    k = P.shape[1]
    srt = np.sort(P, axis=1)[:, ::-1]
    phi = np.column_stack([
        -(P * L).sum(1) / np.log(k),                                  # normalised entropy
        np.log(srt[:, 0]) - np.log(np.maximum(srt[:, 1], 1e-12)),      # log margin
        mass,
        np.full(len(y), np.log(k)),
        np.log1p(ln),
    ])
    return {"L": L, "P": P, "y": y, "phi": phi, "k": k,
            "ids": [r["id"] for r in rs]}


def arms():
    a = sorted(os.path.basename(d) for d in glob.glob(f"{HERE}/results/*") if os.path.isdir(d))
    return [x for x in a if load_full(x, SUITES[0], "test") is not None]


# ---------------------------------------------------------------- pieces
def top_brier(P, y):
    c = P.max(1); o = (P.argmax(1) == y).astype(float)
    return float(np.mean((c - o) ** 2)), c, o


def murphy(c, o, bins=BINS):
    """Binned reliability / resolution / uncertainty of the top-label forecast."""
    edges = np.linspace(0, 1, bins + 1)
    n = len(c); obar = o.mean()
    rel = res = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        if not m.any():
            continue
        w = m.mean()
        rel += w * (c[m].mean() - o[m].mean()) ** 2
        res += w * (o[m].mean() - obar) ** 2
    return float(rel), float(res), float(obar * (1 - obar))


def isotonic(x, o):
    """PAVA: the best monotone map from confidence to P(correct). Fitted on whatever is passed —
    pass the TEST split to get an oracle ceiling, which is the point here."""
    order = np.argsort(x, kind="mergesort")
    v = o[order].astype(float).copy()
    w = np.ones(len(v))
    # pool adjacent violators
    i = 0
    lvl, wt = [], []
    for j in range(len(v)):
        lvl.append(v[j]); wt.append(w[j])
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            b, wb = lvl.pop(), wt.pop()
            a, wa = lvl.pop(), wt.pop()
            lvl.append((a * wa + b * wb) / (wa + wb)); wt.append(wa + wb)
    fit = np.empty(len(v)); p = 0
    for val, ww in zip(lvl, wt):
        fit[p:p + int(ww)] = val; p += int(ww)
    out = np.empty(len(v)); out[order] = fit
    return np.clip(out, 0, 1)


def fit_cond_T(phi_tr, L_tr, y_tr, iters=400, lr=0.25, l2=1e-2):
    """log T_i = w . phi_i, minimising NLL on the training fold. Plain GD with momentum."""
    mu, sd = phi_tr.mean(0), phi_tr.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(y_tr)), (phi_tr - mu) / sd])
    w = np.zeros(Z.shape[1]); vel = np.zeros_like(w)
    n = len(y_tr)
    for _ in range(iters):
        u = np.clip(Z @ w, LOG_T_LO, LOG_T_HI)
        s = np.exp(-u)                                  # s = 1/T
        Zl = L_tr * s[:, None]
        Zl = Zl - Zl.max(1, keepdims=True)
        P = np.exp(Zl); P /= P.sum(1, keepdims=True)
        Ly = L_tr[np.arange(n), y_tr]
        EL = (P * L_tr).sum(1)
        g_u = s * (Ly - EL)                             # dNLL/du
        g = Z.T @ g_u / n + l2 * w
        vel = 0.9 * vel - lr * g
        w = w + vel
    return {"w": w, "mu": mu, "sd": sd}


def apply_cond_T(model, phi):
    Z = np.column_stack([np.ones(len(phi)), (phi - model["mu"]) / model["sd"]])
    return np.exp(np.clip(Z @ model["w"], LOG_T_LO, LOG_T_HI))


def soft(L, T):
    Z = L / np.asarray(T).reshape(-1, 1)
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


# ---------------------------------------------------------------- main
def main():
    A = arms()
    out = {"ceiling": [], "transfer": [], "ensemble": []}
    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)

    # ---------- 1 + 2. how much scale is even on the table ----------
    emit("== 1. THE SCALE BUDGET: what ANY post-hoc recalibration could win ==")
    emit("Top-label Brier. 'oracle' = isotonic fitted ON THE TEST SET — not a method, a ceiling.")
    emit("budget = raw - oracle is the entire scale term; 'floor' (= oracle) is ranking-bound and")
    emit("only training moves it. 'T-in cap' = share of the budget an in-domain temperature takes.")
    emit()
    hdr = (f"{'arm':20s} {'suite':10s} {'brier':>6s} {'oracle':>6s} {'budget':>6s} "
           f"{'T-in':>6s} {'cap%':>5s} | {'REL':>6s} {'RES':>6s} {'AURC':>6s}")
    emit(hdr); emit("-" * len(hdr))
    for a in A:
        for s in SUITES:
            d = load_full(a, s, "test")
            dc = load_full(a, s, "calib")
            T_in = M.fit_T([(dc["L"], dc["y"])])
            b_raw, c, o = top_brier(d["P"], d["y"])
            b_in, _, _ = top_brier(soft(d["L"], np.full(len(d["y"]), T_in)), d["y"])
            iso = isotonic(c, o)
            b_or = float(np.mean((iso - o) ** 2))
            rel, res, unc = murphy(c, o)
            budget = b_raw - b_or
            cap = (b_raw - b_in) / budget if budget > 1e-9 else float("nan")
            au = M.aurc(d["P"], d["y"])
            out["ceiling"].append({"arm": a, "suite": s, "brier": b_raw, "oracle": b_or,
                                   "budget": budget, "brier_Tin": b_in, "cap": cap,
                                   "rel": rel, "res": res, "unc": unc, "aurc": au})
            emit(f"{a:20s} {s:10s} {b_raw:6.3f} {b_or:6.3f} {budget:6.3f} "
                 f"{b_in:6.3f} {100*cap:4.0f}% | {rel:6.3f} {res:6.3f} {au:6.3f}")
        emit()
    cs = out["ceiling"]
    emit(f"pooled: mean Brier {np.mean([r['brier'] for r in cs]):.3f}, "
         f"mean oracle floor {np.mean([r['oracle'] for r in cs]):.3f}, "
         f"mean scale budget {np.mean([r['budget'] for r in cs]):.3f} "
         f"({100*np.mean([r['budget'] for r in cs])/np.mean([r['brier'] for r in cs]):.0f}% of Brier)")
    emit()

    # ---------- 3. conditional temperature, leave-one-suite-out ----------
    emit("== 3. CONDITIONAL TEMPERATURE, LEAVE-ONE-SUITE-OUT ==")
    emit("log T = w . [entropy, margin, label_mass, log k, log len], fitted on the OTHER three")
    emit("suites' calib splits, applied to a suite it has never seen. Compared against the same")
    emit("fold's global scalar T-loso. A per-item T can reorder, so AURC is in play too.")
    emit()
    hdr = (f"{'arm':20s} {'suite':10s} | {'ECE raw':>7s} {'glob':>6s} {'cond':>6s} "
           f"{'d cond vs glob [95% CI]':>28s} | {'AURC raw':>8s} {'cond':>6s} {'dAURC':>18s}")
    emit(hdr); emit("-" * len(hdr))
    tal = {"ece_better": 0, "ece_worse": 0, "aurc_better": 0, "aurc_worse": 0, "cells": 0}
    for a in A:
        for s in SUITES:
            te = load_full(a, s, "test")
            tr = [load_full(a, o, "calib") for o in SUITES if o != s]
            phi_tr = np.vstack([t["phi"] for t in tr])
            y_tr = np.concatenate([t["y"] for t in tr])
            kmax = max(t["L"].shape[1] for t in tr)
            # pad logit rows to a common width with -inf-ish so one model spans mixed-k tasks
            L_tr = np.full((len(y_tr), kmax), -60.0)
            p = 0
            for t in tr:
                L_tr[p:p + len(t["y"]), :t["L"].shape[1]] = t["L"]; p += len(t["y"])
            mdl = fit_cond_T(phi_tr, L_tr, y_tr)
            T_glob = M.fit_T([(t["L"], t["y"]) for t in tr])
            T_cond = apply_cond_T(mdl, te["phi"])
            P_raw = te["P"]
            P_g = soft(te["L"], np.full(len(te["y"]), T_glob))
            P_c = soft(te["L"], T_cond)
            y = te["y"]
            e_raw, e_g, e_c = M.ece(P_raw, y), M.ece(P_g, y), M.ece(P_c, y)
            a_raw, a_c = M.aurc(P_raw, y), M.aurc(P_c, y)
            de, da = [], []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y)); yy = y[i]
                de.append(M.ece(P_c[i], yy) - M.ece(P_g[i], yy))
                da.append(M.aurc(P_c[i], yy) - M.aurc(P_raw[i], yy))
            def fm(v):
                lo, hi = np.percentile(v, [2.5, 97.5]); m = float(np.mean(v))
                vd = "better" if hi < 0 else "WORSE" if lo > 0 else "n.s."
                return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] {vd:6s}", vd
            fe, ve = fm(de); fa, va = fm(da)
            tal["cells"] += 1
            tal["ece_better"] += ve == "better"; tal["ece_worse"] += ve == "WORSE"
            tal["aurc_better"] += va == "better"; tal["aurc_worse"] += va == "WORSE"
            out["transfer"].append({"arm": a, "suite": s, "ece_raw": e_raw, "ece_glob": e_g,
                                    "ece_cond": e_c, "d_ece": float(np.mean(de)),
                                    "d_ece_verdict": ve, "aurc_raw": a_raw, "aurc_cond": a_c,
                                    "d_aurc": float(np.mean(da)), "d_aurc_verdict": va,
                                    "T_glob": T_glob, "T_cond_med": float(np.median(T_cond)),
                                    "w": mdl["w"].tolist()})
            emit(f"{a:20s} {s:10s} | {e_raw:7.3f} {e_g:6.3f} {e_c:6.3f} {fe:>28s} | "
                 f"{a_raw:8.4f} {a_c:6.4f} {fa:>18s}")
        emit()
    emit(f"cells={tal['cells']}  ECE cond-vs-global: better {tal['ece_better']}, "
         f"WORSE {tal['ece_worse']}  |  AURC cond-vs-raw: better {tal['aurc_better']}, "
         f"WORSE {tal['aurc_worse']}  (rest n.s.)")
    emit()

    # ---------- 4. the free ensemble ----------
    emit("== 4. FORMAT AGREEMENT AS A CONFIDENCE SIGNAL (two passes we already paid for) ==")
    emit("For arms run in both readouts: does combining them rank better than the name format")
    emit("alone? 'mean' averages the two distributions; 'gated' zeroes confidence on disagreement.")
    emit()
    hdr = f"{'arm':14s} {'suite':10s} {'AURC name':>9s} {'mean':>7s} {'gated':>7s} {'agree%':>7s} {'acc|agree':>9s}"
    emit(hdr); emit("-" * len(hdr))
    for base in ["Qwen3.5-2B", "Qwen3.5-4B", "smart"]:
        an, al = f"{base}-name", f"{base}-letter"
        if an not in A or al not in A:
            continue
        for s in SUITES:
            n_, l_ = load_full(an, s, "test"), load_full(al, s, "test")
            y = n_["y"]
            agree = (n_["P"].argmax(1) == l_["P"].argmax(1))
            Pm = (n_["P"] + l_["P"]) / 2
            conf_g = n_["P"].max(1) * np.where(agree, 1.0, 0.0)
            # gated ranking: keep argmax from the name arm, push disagreements to the bottom
            ordg = np.argsort(-(conf_g + 1e-9 * n_["P"].max(1)))
            err = (n_["P"].argmax(1)[ordg] != y[ordg]).astype(float)
            aurc_g = float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))
            acc_ag = float((n_["P"].argmax(1)[agree] == y[agree]).mean()) if agree.any() else float("nan")
            row = {"arm": base, "suite": s, "aurc_name": M.aurc(n_["P"], y),
                   "aurc_mean": M.aurc(Pm, y), "aurc_gated": aurc_g,
                   "agree": float(agree.mean()), "acc_given_agree": acc_ag}
            out["ensemble"].append(row)
            emit(f"{base:14s} {s:10s} {row['aurc_name']:9.4f} {row['aurc_mean']:7.4f} "
                 f"{aurc_g:7.4f} {100*row['agree']:6.0f}% {acc_ag:9.3f}")
        emit()

    json.dump(out, open(f"{HERE}/results/transfer.json", "w"), indent=1)
    open(f"{HERE}/results/transfer.txt", "w").write("\n".join(lines) + "\n")
    print("wrote results/transfer.{json,txt}")


if __name__ == "__main__":
    main()
