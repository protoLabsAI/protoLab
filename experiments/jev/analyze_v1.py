#!/usr/bin/env python3
"""Rung 1 analysis — 22 tasks, 6 families, leave-one-FAMILY-out. numpy only.

Everything before this ran on four suites, which is four folds: enough to see the direction of
an effect, not enough to claim transfer. This is the version of the analysis the claims actually
need. It writes v1 outputs under new names; the v0 reference files are left untouched so the
published v0 numbers stay reproducible.

  1. board            accuracy + AURC per arm, macro-averaged within family
  2. cascade          does the headline survive 22 tasks — a 4B deferring to the 27B
  3. LOFO calibration the real test: fit a temperature on five families, apply to the sixth
  4. scale budget     oracle recalibration ceiling per family
  5. verifier         Finding 12 across the full set

Usage:  python analyze_v1.py            # writes results/v1-analysis.{txt,json}
"""
import glob, json, os
import numpy as np
import metrics as M
import calib_transfer as C

HERE = os.path.dirname(os.path.abspath(__file__))
B = 1000
rng = np.random.default_rng(20260921)
REF = "smart-name"
DECIDER = "Qwen3.5-4B-name"
DEFER_GRID = [0.0, 0.10, 0.20, 0.30, 0.50]


def roster():
    out = {}
    for f in sorted(glob.glob(f"{HERE}/data/*.meta.json")):
        m = json.load(open(f))
        out[m["suite"]] = m.get("family", "unknown")
    return out


def arms(suites):
    a = sorted(os.path.basename(d) for d in glob.glob(f"{HERE}/results/*") if os.path.isdir(d))
    return [x for x in a
            if all(os.path.exists(f"{HERE}/results/{x}/{s}.test.jsonl") for s in suites)]


def cell(arm, suite, split="test"):
    d = C.load_full(arm, suite, split)
    if d is None:
        return None
    P = d["P"]
    pick = P.argmax(1)
    return {"P": P, "L": d["L"], "y": d["y"], "phi": d["phi"], "pick": pick,
            "corr": (pick == d["y"]).astype(float), "conf": P.max(1), "k": d["k"]}


def aurc_from(conf, corr):
    order = np.argsort(-conf)
    err = 1.0 - corr[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def casc(conf_a, corr_a, corr_f, d):
    n = len(corr_a)
    k = n - int(round(d * n))
    order = np.argsort(-conf_a)
    return float((corr_a[order[:k]].sum() + corr_f[order[k:]].sum()) / n)


def main():
    R = roster()
    suites = sorted(R)
    fams = sorted(set(R.values()))
    A = arms(suites)
    lines, out = [], {"roster": R}
    def emit(s=""):
        lines.append(s); print(s, flush=True)

    emit(f"== RUNG 1: {len(suites)} tasks, {len(fams)} families, {len(A)} arms ==")
    emit(f"families: " + "  ".join(f"{f}({sum(1 for v in R.values() if v == f)})" for f in fams))
    emit()

    TEST = {a: {s: cell(a, s) for s in suites} for a in A}
    CAL = {a: {s: cell(a, s, "calib") for s in suites} for a in A}

    # ---------------- 1. board ----------------
    emit("== 1. BOARD — accuracy and AURC, macro-averaged within family ==")
    emit("AURC lower is better. 'all' is the macro-average over the 22 tasks.")
    emit()
    hdr = f"{'arm':20s} {'metric':6s} " + " ".join(f"{f[:9]:>9s}" for f in fams) + f" {'all':>7s}"
    emit(hdr); emit("-" * len(hdr))
    for a in A:
        accs, aus = {}, {}
        for f in fams:
            ss = [s for s in suites if R[s] == f]
            accs[f] = np.mean([TEST[a][s]["corr"].mean() for s in ss])
            aus[f] = np.mean([aurc_from(TEST[a][s]["conf"], TEST[a][s]["corr"]) for s in ss])
        all_acc = np.mean([TEST[a][s]["corr"].mean() for s in suites])
        all_au = np.mean([aurc_from(TEST[a][s]["conf"], TEST[a][s]["corr"]) for s in suites])
        out.setdefault("board", []).append({"arm": a, "acc": accs, "aurc": aus,
                                            "acc_all": float(all_acc), "aurc_all": float(all_au)})
        emit(f"{a:20s} {'acc':6s} " + " ".join(f"{accs[f]:9.3f}" for f in fams) + f" {all_acc:7.3f}")
        emit(f"{'':20s} {'AURC':6s} " + " ".join(f"{aus[f]:9.4f}" for f in fams) + f" {all_au:7.4f}")
    emit()

    # ---------------- 2. cascade ----------------
    emit(f"== 2. CASCADE at 22 tasks — {DECIDER} defers to {REF} ==")
    emit("Macro-average over 22 tasks. Paired bootstrap resamples items within each task.")
    emit()
    hdr = f"{'arm':20s} {'defer%':>6s} {'pooled acc':>10s} {'d vs 27B alone':>26s}"
    emit(hdr); emit("-" * len(hdr))
    ref_pool = float(np.mean([TEST[REF][s]["corr"].mean() for s in suites]))
    emit(f"{REF + ' alone':20s} {'100':>6s} {ref_pool:10.3f} {'--':>26s}")
    for a in [DECIDER, "embed-head", "Qwen3.5-2B-name"]:
        if a not in A:
            continue
        for d in DEFER_GRID:
            acc = float(np.mean([casc(TEST[a][s]["conf"], TEST[a][s]["corr"],
                                      TEST[REF][s]["corr"], d) for s in suites]))
            if d in (0.0, 0.20, 0.30, 0.50):
                deltas = []
                for _ in range(B):
                    ds = []
                    for s in suites:
                        n = len(TEST[a][s]["corr"])
                        i = rng.integers(0, n, n)
                        ds.append(casc(TEST[a][s]["conf"][i], TEST[a][s]["corr"][i],
                                       TEST[REF][s]["corr"][i], d) - TEST[REF][s]["corr"][i].mean())
                    deltas.append(np.mean(ds))
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                v = "BEATS 27B" if lo > 0 else "worse" if hi < 0 else "n.s."
                txt = f"{np.mean(deltas):+.3f} [{lo:+.3f},{hi:+.3f}] {v}"
                out.setdefault("cascade", []).append(
                    {"arm": a, "defer": d, "acc": acc, "delta": float(np.mean(deltas)),
                     "ci": [float(lo), float(hi)], "verdict": v})
            else:
                txt = ""
            emit(f"{a:20s} {int(d*100):5d}% {acc:10.3f} {txt:>26s}")
        emit()

    # ---------------- 3. leave-one-FAMILY-out calibration ----------------
    emit("== 3. LEAVE-ONE-FAMILY-OUT CALIBRATION — the test rung 1 exists for ==")
    emit("T-lofo: one temperature fitted on the calib splits of the OTHER FIVE families, applied")
    emit("to every task in the held-out family. This is the claim a general decision model makes.")
    emit("cond: the same, with a per-item temperature from decode-time features.")
    emit()
    hdr = (f"{'arm':20s} {'held-out family':16s} {'ECE raw':>8s} {'T-lofo':>7s} {'cond':>7s} "
           f"{'d lofo vs raw':>24s}")
    emit(hdr); emit("-" * len(hdr))
    tal = {"better": 0, "worse": 0, "ns": 0}
    for a in A:
        for f in fams:
            held = [s for s in suites if R[s] == f]
            train = [s for s in suites if R[s] != f]
            T = M.fit_T([(CAL[a][s]["L"], CAL[a][s]["y"]) for s in train])
            phi_tr = np.vstack([CAL[a][s]["phi"] for s in train])
            y_tr = np.concatenate([CAL[a][s]["y"] for s in train])
            kmax = max(CAL[a][s]["L"].shape[1] for s in train)
            L_tr = np.full((len(y_tr), kmax), -60.0)
            p = 0
            for s in train:
                L = CAL[a][s]["L"]
                L_tr[p:p + len(L), :L.shape[1]] = L; p += len(L)
            mdl = C.fit_cond_T(phi_tr, L_tr, y_tr)
            e_raw = np.mean([M.ece(TEST[a][s]["P"], TEST[a][s]["y"]) for s in held])
            e_T = np.mean([M.ece(C.soft(TEST[a][s]["L"], np.full(len(TEST[a][s]["y"]), T)),
                                 TEST[a][s]["y"]) for s in held])
            e_c = np.mean([M.ece(C.soft(TEST[a][s]["L"], C.apply_cond_T(mdl, TEST[a][s]["phi"])),
                                 TEST[a][s]["y"]) for s in held])
            d = []
            for _ in range(B):
                ds = []
                for s in held:
                    n = len(TEST[a][s]["y"]); i = rng.integers(0, n, n)
                    yy = TEST[a][s]["y"][i]
                    ds.append(M.ece(C.soft(TEST[a][s]["L"][i], np.full(n, T)), yy)
                              - M.ece(TEST[a][s]["P"][i], yy))
                d.append(np.mean(ds))
            lo, hi = np.percentile(d, [2.5, 97.5])
            v = "better" if hi < 0 else "WORSE" if lo > 0 else "n.s."
            tal["better" if v == "better" else "worse" if v == "WORSE" else "ns"] += 1
            out.setdefault("lofo", []).append(
                {"arm": a, "family": f, "ece_raw": float(e_raw), "ece_lofo": float(e_T),
                 "ece_cond": float(e_c), "T": T, "delta": float(np.mean(d)),
                 "ci": [float(lo), float(hi)], "verdict": v})
            emit(f"{a:20s} {f:16s} {e_raw:8.3f} {e_T:7.3f} {e_c:7.3f} "
                 f"{np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] {v:6s}")
        emit()
    emit(f"cells={sum(tal.values())}  T-lofo: better {tal['better']}, WORSE {tal['worse']}, "
         f"n.s. {tal['ns']}")
    emit()

    # ---------------- 4. scale budget per family ----------------
    emit("== 4. SCALE BUDGET per family (oracle isotonic on test = the ceiling) ==")
    emit()
    hdr = f"{'arm':20s} {'family':16s} {'Brier':>7s} {'oracle':>7s} {'budget':>7s} {'% of Brier':>10s}"
    emit(hdr); emit("-" * len(hdr))
    for a in [REF, DECIDER, "Qwen3.5-0.8B-name"]:
        if a not in A:
            continue
        for f in fams:
            held = [s for s in suites if R[s] == f]
            br, orc = [], []
            for s in held:
                b, c, o = C.top_brier(TEST[a][s]["P"], TEST[a][s]["y"])
                br.append(b); orc.append(float(np.mean((C.isotonic(c, o) - o) ** 2)))
            b, o = float(np.mean(br)), float(np.mean(orc))
            out.setdefault("budget", []).append({"arm": a, "family": f, "brier": b, "oracle": o,
                                                 "budget": b - o})
            emit(f"{a:20s} {f:16s} {b:7.3f} {o:7.3f} {b-o:7.3f} {100*(b-o)/max(b,1e-9):9.0f}%")
        emit()

    # ---------------- 5. verifier at scale ----------------
    emit("== 5. VERIFIER vs SELF-CONFIDENCE across all 22 tasks ==")
    emit(f"Ranking {DECIDER}'s answers by P_verifier(the option it chose), vs its own confidence.")
    emit()
    hdr = f"{'verifier':20s} {'family':16s} {'self AURC':>9s} {'verifier':>9s} {'delta':>20s}"
    emit(hdr); emit("-" * len(hdr))
    for v in ["embed-head", "Qwen3.5-2B-name", REF]:
        if v not in A or v == DECIDER:
            continue
        for f in fams:
            held = [s for s in suites if R[s] == f]
            au_s, au_v, ds = [], [], []
            for s in held:
                t = TEST[DECIDER][s]
                vc = TEST[v][s]["P"][np.arange(len(t["pick"])), t["pick"]]
                au_s.append(aurc_from(t["conf"], t["corr"]))
                au_v.append(aurc_from(vc, t["corr"]))
            for _ in range(B):
                dd = []
                for s in held:
                    t = TEST[DECIDER][s]
                    vc = TEST[v][s]["P"][np.arange(len(t["pick"])), t["pick"]]
                    n = len(t["corr"]); i = rng.integers(0, n, n)
                    dd.append(aurc_from(vc[i], t["corr"][i]) - aurc_from(t["conf"][i], t["corr"][i]))
                ds.append(np.mean(dd))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            verdict = "better" if hi < 0 else "WORSE" if lo > 0 else "n.s."
            out.setdefault("verifier", []).append(
                {"verifier": v, "family": f, "aurc_self": float(np.mean(au_s)),
                 "aurc_verifier": float(np.mean(au_v)), "delta": float(np.mean(ds)),
                 "ci": [float(lo), float(hi)], "verdict": verdict})
            emit(f"{v:20s} {f:16s} {np.mean(au_s):9.4f} {np.mean(au_v):9.4f} "
                 f"{np.mean(ds):+.4f} [{lo:+.4f},{hi:+.4f}] {verdict}")
        emit()

    json.dump(out, open(f"{HERE}/results/v1-analysis.json", "w"), indent=1)
    open(f"{HERE}/results/v1-analysis.txt", "w").write("\n".join(lines) + "\n")
    print("wrote results/v1-analysis.{txt,json}")


if __name__ == "__main__":
    main()
