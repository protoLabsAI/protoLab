#!/usr/bin/env python3
"""Supervised baseline: Qwen3-Embedding-0.6B (:8001) + softmax-regression head per suite.

Unlike every other arm this one SEES LABELS (the train split). It is the "boring local
classifier" bar: if a 0.6B embedder plus a few hundred labels beats zero-shot decoding, the
zero-shot story has to justify itself on no-labels-needed, not on quality.
Same output format as decide.py, so metrics.py scores it identically.
"""
import json, os, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://localhost:8001/v1/embeddings"; MODEL = "Qwen/Qwen3-Embedding-0.6B"
SUITES = ["injection", "sentiment", "routing", "langid"]

def embed(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        body = {"model": MODEL, "input": [t[:2000] for t in texts[i:i+bs]]}
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}), timeout=120))
        out += [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]
    E = np.array(out, float); return E / np.linalg.norm(E, axis=1, keepdims=True)

def train_softmax(X, y, k, l2=1e-3, iters=600, lr=0.5):
    W = np.zeros((X.shape[1], k)); b = np.zeros(k); Y = np.eye(k)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(y)
        W -= lr * (X.T @ G + l2 * W); b -= lr * G.sum(0)
    return W, b

def main():
    tag = "embed-head"; out = f"{HERE}/results/{tag}"; os.makedirs(out, exist_ok=True)
    for s in SUITES:
        meta = json.load(open(f"{HERE}/data/{s}.meta.json")); k = len(meta["options"])
        split = {sp: [json.loads(l) for l in open(f"{HERE}/data/{s}.{sp}.jsonl")] for sp in ("train", "calib", "test")}
        ys = {sp: np.array([meta["options"].index(r["label"]) for r in rows]) for sp, rows in split.items()}
        Xs = {sp: embed([r["text"] for r in rows]) for sp, rows in split.items()}
        W, b = train_softmax(Xs["train"], ys["train"], k)
        for sp in ("test", "calib"):
            Z = Xs[sp] @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            with open(f"{out}/{s}.{sp}.jsonl", "w") as f:
                for r, p, y in zip(split[sp], P, ys[sp]):
                    f.write(json.dumps({"id": r["id"], "y": int(y), "probs": p.tolist(), "label_mass": 1.0, "missing": 0}) + "\n")
        acc = float((((Xs["test"] @ W + b).argmax(1)) == ys["test"]).mean())
        print(f"{tag:20s} {s:10s} train={len(ys['train']):4d} test acc={acc:.3f}", flush=True)

if __name__ == "__main__":
    main()
