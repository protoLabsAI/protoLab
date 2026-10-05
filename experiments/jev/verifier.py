#!/usr/bin/env python3
"""Self-confidence vs an INDEPENDENT VERIFIER as the deferral signal.

Surveying what people actually build with Jev (2026-09-20) turned up a scope gap in this
experiment. The flagship pattern — TypeSafe's own workflows, OpenRouter's "draft-verify-escalate"
cookbook, both community use-case catalogues — does not use the decision model as the DECIDER
that defers on its own confidence. It uses it as a VERIFIER sitting between a cheap draft and a
frontier escalation, judging someone else's answer.

Those are different signals and they can behave very differently. A decider's own confidence is
correlated with its own errors by construction; an independent verifier's is not.

Testable with data already in hand, because every arm produced a full distribution over the same
items: for each item, arm A picks option j, and the verifier's probability on THAT option,
P_V(j), is exactly "an independent model's opinion of A's answer".

Gates compared, all ranking the same arm-A answers:
  self      A's own max probability                 (what this experiment measured)
  verifier  P_V(A's chosen option)                  (what the field actually deploys)
  min       min(self, verifier)                     (agreement-weighted)

Usage:  python verifier.py [--decider ARM] [--verifiers ARM ...]
"""
import argparse, json, os
import numpy as np
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
B = 2000
rng = np.random.default_rng(20260920)


def rank_aurc(conf, correct):
    """AURC of a ranking induced by an arbitrary confidence signal over fixed answers."""
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def sel_at(conf, correct, cov):
    k = max(1, int(round(cov * len(correct))))
    return float(correct[np.argsort(-conf)[:k]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", default="Qwen3.5-4B-name")
    ap.add_argument("--verifiers", nargs="*",
                    default=["embed-head", "Qwen3.5-2B-name", "Qwen3.5-0.8B-name", "smart-name"])
    ap.add_argument("--suites", nargs="*", default=None)
    a = ap.parse_args()
    suites = a.suites or M.SUITES
    out = []

    print(f"decider = {a.decider}   (the arm whose answers are being ranked)")
    print("AURC of the deferral ranking, lower is better. 'self' is this experiment's signal;")
    print("'verifier' is P_verifier(the option the decider chose) — the deployed pattern.\n")
    hdr = (f"{'suite':10s} {'verifier':18s} {'acc':>6s} {'self':>7s} {'verifier':>8s} "
           f"{'min':>7s} {'d verif vs self [95% CI]':>28s}")
    print(hdr); print("-" * len(hdr))
    for s in suites:
        dA = M.load(a.decider, s, "test")
        if dA is None:
            continue
        LA, y, _ = dA
        PA = M.softmax(LA)
        pick = PA.argmax(1)
        correct = (pick == y).astype(float)
        self_conf = PA.max(1)
        for v in a.verifiers:
            if v == a.decider:
                continue
            dV = M.load(v, s, "test")
            if dV is None:
                continue
            PV = M.softmax(dV[0])
            ver_conf = PV[np.arange(len(pick)), pick]
            mn = np.minimum(self_conf, ver_conf)
            au_s, au_v, au_m = (rank_aurc(self_conf, correct), rank_aurc(ver_conf, correct),
                                rank_aurc(mn, correct))
            d = []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y))
                d.append(rank_aurc(ver_conf[i], correct[i]) - rank_aurc(self_conf[i], correct[i]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            verdict = "better" if hi < 0 else "WORSE" if lo > 0 else "n.s."
            out.append({"suite": s, "decider": a.decider, "verifier": v, "acc": float(correct.mean()),
                        "aurc_self": au_s, "aurc_verifier": au_v, "aurc_min": au_m,
                        "delta": float(np.mean(d)), "ci": [float(lo), float(hi)],
                        "verdict": verdict,
                        "sel90_self": sel_at(self_conf, correct, 0.9),
                        "sel90_verifier": sel_at(ver_conf, correct, 0.9),
                        "sel90_min": sel_at(mn, correct, 0.9)})
            print(f"{s:10s} {v:18s} {correct.mean():6.3f} {au_s:7.4f} {au_v:8.4f} {au_m:7.4f} "
                  f"{np.mean(d):+.4f} [{lo:+.4f},{hi:+.4f}] {verdict:6s}")
        print()

    json.dump(out, open(f"{HERE}/results/verifier.json", "w"), indent=1)
    n_better = sum(r["verdict"] == "better" for r in out)
    n_worse = sum(r["verdict"] == "WORSE" for r in out)
    print(f"cells={len(out)}  verifier beats self-confidence: {n_better}, loses: {n_worse}, "
          f"n.s.: {len(out)-n_better-n_worse}")
    best = min(out, key=lambda r: r["aurc_min"])
    print(f"\nbest combined gate: {best['suite']} + {best['verifier']} -> AURC {best['aurc_min']:.4f} "
          f"vs self {best['aurc_self']:.4f}")
    print("wrote results/verifier.json")


if __name__ == "__main__":
    main()
