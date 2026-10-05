#!/usr/bin/env python3
"""Re-fit the embed-head baseline and SAVE the weights (embed_head.py trains and discards them).

Verifies before it writes: the recomputed test probabilities must match the committed
results/embed-head/*.jsonl. If the embedding lane has drifted since the original run, the
published numbers and the published weights would disagree — so this reports the mismatch and
refuses to write rather than silently shipping a second, different baseline.

  python export_embed_head.py            # needs the embed lane on :8001
"""
import json, os, sys
import numpy as np
import embed_head as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/baselines/embed-head"
TOL = 1e-6


def main():
    os.makedirs(OUT, exist_ok=True)
    worst = 0.0
    report = []
    for s in E.SUITES:
        meta = json.load(open(f"{HERE}/data/{s}.meta.json"))
        k = len(meta["options"])
        split = {sp: [json.loads(l) for l in open(f"{HERE}/data/{s}.{sp}.jsonl")]
                 for sp in ("train", "calib", "test")}
        ys = {sp: np.array([meta["options"].index(r["label"]) for r in rows])
              for sp, rows in split.items()}
        Xs = {sp: E.embed([r["text"] for r in rows]) for sp, rows in split.items()}
        W, b = E.train_softmax(Xs["train"], ys["train"], k)

        Z = Xs["test"] @ W + b
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        ref = np.array([json.loads(l)["probs"]
                        for l in open(f"{HERE}/results/embed-head/{s}.test.jsonl")], float)
        d = float(np.abs(P - ref).max())
        worst = max(worst, d)
        report.append((s, d, float((P.argmax(1) == ys["test"]).mean())))
        np.savez(f"{OUT}/{s}.npz", W=W.astype(np.float32), b=b.astype(np.float32),
                 options=np.array(meta["options"], dtype=object),
                 embed_model=E.MODEL, n_train=len(ys["train"]))

    print(f"{'suite':10s} {'max |dP| vs committed':>22s} {'test acc':>9s}")
    for s, d, acc in report:
        print(f"{s:10s} {d:22.2e} {acc:9.3f}")
    if worst > TOL:
        print(f"\nMISMATCH: worst {worst:.2e} > {TOL:.0e} — the embedding lane has drifted since "
              f"the original run. Weights written to {OUT} but they do NOT reproduce the "
              f"published probabilities; do not publish both without re-running the scoreboard.",
              file=sys.stderr)
        return 1
    print(f"\nreproduces committed probabilities (worst {worst:.2e}); weights in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
