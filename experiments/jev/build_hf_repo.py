#!/usr/bin/env python3
"""Assemble the deferral-bench-v0 HuggingFace dataset repo from this experiment's outputs.

Builds a staging tree only — it never touches the network and never pushes. Publishing is a
separate, explicit step.

    python build_hf_repo.py [--out hf/deferral-bench-v0]

What ships:  item ids, gold labels, SHA-256 of each text, and the per-item probability vectors
             from every arm, plus the embed-head baseline weights and the scorers.
What does not: the source texts. Four sources, four licenses — `rebuild.py` reconstructs them
             from upstream and verifies every hash.
"""
import argparse, hashlib, json, os, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["test", "calib"]


def suites():
    import glob as _g
    return sorted(json.load(open(f))["suite"] for f in _g.glob(f"{HERE}/data/*.meta.json"))


def family(s):
    return json.load(open(f"{HERE}/data/{s}.meta.json")).get("family", "unknown")

SOURCE = {
    "injection": {"hf_dataset": "deepset/prompt-injections",
                  "note": "train+test pooled (662 rows total), then re-split by us"},
    "sentiment": {"hf_dataset": "stanfordnlp/sst2",
                  "note": "validation split — sst2's test split is unlabelled"},
    "routing":   {"hf_dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                  "note": "train split; label = the dataset's own top-level `category`"},
    "langid":    {"hf_dataset": "papluca/language-identification",
                  "note": "test split; ISO codes mapped to English language names"},
}

ARMS = {
    "smart-name": {
        "model": "Qwen3.8-27B-NVFP4 (+ MTP K=3)", "params": "27B", "precision": "NVFP4",
        "readout": "name", "where": "vLLM lane on an RTX PRO 6000 Blackwell",
        "sees_labels": False, "note": "the frontier reference arm; a live production lane"},
    "smart-letter": {
        "model": "Qwen3.8-27B-NVFP4 (+ MTP K=3)", "params": "27B", "precision": "NVFP4",
        "readout": "letter", "where": "vLLM lane on an RTX PRO 6000 Blackwell",
        "sees_labels": False, "note": "same model, MMLU-style A)/B) readout"},
    "Qwen3.5-4B-name": {
        "model": "Qwen/Qwen3.5-4B", "params": "4B", "precision": "bf16", "readout": "name",
        "where": "CPU (transformers)", "sees_labels": False,
        "note": "the arm that matches the 27B under a 20% deferral budget"},
    "Qwen3.5-4B-letter": {
        "model": "Qwen/Qwen3.5-4B", "params": "4B", "precision": "bf16", "readout": "letter",
        "where": "CPU (transformers)", "sees_labels": False},
    "Qwen3.5-2B-name": {
        "model": "Qwen/Qwen3.5-2B", "params": "2B", "precision": "bf16", "readout": "name",
        "where": "CPU (transformers)", "sees_labels": False},
    "Qwen3.5-2B-letter": {
        "model": "Qwen/Qwen3.5-2B", "params": "2B", "precision": "bf16", "readout": "letter",
        "where": "CPU (transformers)", "sees_labels": False,
        "note": "the cautionary arm: ECE 0.006 after temperature scaling at 10% accuracy on "
                "langid, with confidence ANTI-correlated with correctness"},
    "Qwen3.5-0.8B-name": {
        "model": "Qwen/Qwen3.5-0.8B", "params": "0.8B", "precision": "bf16", "readout": "name",
        "where": "CPU (transformers)", "sees_labels": False,
        "note": "below the capability floor; only 31% of raw next-token mass lands in-schema"},
    "embed-head": {
        "model": "Qwen/Qwen3-Embedding-0.6B + softmax head", "params": "0.6B",
        "precision": "bf16", "readout": "n/a (supervised head)", "where": "vLLM embedding lane",
        "sees_labels": True,
        "note": "the only arm trained on labels (162-1000 per suite); weights in baselines/"},
}

REFERENCE = ["deferral.txt", "deferral.json", "transfer.txt", "transfer.json",
             "bootstrap.txt", "summary.json", "v1-analysis.txt", "v1-analysis.json",
             "verifier.json"]


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/hf/deferral-bench-v1")
    a = ap.parse_args()
    out = a.out
    for sub in ["manifest", "meta", "probs", "reference", "baselines"]:
        os.makedirs(f"{out}/{sub}", exist_ok=True)

    # ---- manifests + meta ----
    counts = {}
    SUITES = suites()
    for s in SUITES:
        meta = json.load(open(f"{HERE}/data/{s}.meta.json"))
        for sp in SPLITS:
            rows = [json.loads(l) for l in open(f"{HERE}/data/{s}.{sp}.jsonl")]
            df = pd.DataFrame([{
                "id": r["id"], "suite": s, "split": sp, "position": i,
                "label": r["label"], "label_idx": meta["options"].index(r["label"]),
                "text_sha256": sha(r["text"]), "text_len": len(r["text"]),
            } for i, r in enumerate(rows)])
            df.to_parquet(f"{out}/manifest/{s}.{sp}.parquet", index=False)
            counts[f"{s}.{sp}"] = len(df)
        json.dump({**meta, "family": family(s),
                   "source": SOURCE.get(s, meta.get("source", {"note": "see rebuild.py"})),
                   "k": len(meta["options"])},
                  open(f"{out}/meta/{s}.json", "w"), indent=1)

    # ---- probability vectors ----
    n_vec = 0
    for arm in ARMS:
        src = f"{HERE}/results/{arm}"
        if not os.path.isdir(src):
            print(f"!! missing arm {arm}, skipping")
            continue
        os.makedirs(f"{out}/probs/{arm}", exist_ok=True)
        for s in SUITES:
            for sp in SPLITS:
                p = f"{src}/{s}.{sp}.jsonl"
                if not os.path.exists(p):
                    continue
                rs = [json.loads(l) for l in open(p)]
                df = pd.DataFrame({
                    "id": [r["id"] for r in rs],
                    "y": [r["y"] for r in rs],
                    "probs": [np.asarray(r["probs"], dtype=np.float64) for r in rs],
                    "label_mass": [float(r.get("label_mass", 1.0)) for r in rs],
                    "missing": [int(r.get("missing", 0)) for r in rs],
                })
                df.to_parquet(f"{out}/probs/{arm}/{s}.{sp}.parquet", index=False)
                n_vec += len(df)

    # ---- baselines, reference outputs, scorers ----
    if os.path.isdir(f"{HERE}/baselines/embed-head"):
        shutil.copytree(f"{HERE}/baselines/embed-head", f"{out}/baselines/embed-head",
                        dirs_exist_ok=True)
    for f in REFERENCE:
        if os.path.exists(f"{HERE}/results/{f}"):
            shutil.copy(f"{HERE}/results/{f}", f"{out}/reference/{f}")
    for f in ["scorer.py", "rebuild.py", "README.md"]:
        shutil.copy(f"{HERE}/hf_assets/{f}", f"{out}/{f}")

    json.dump({"arms": ARMS, "families": {s: family(s) for s in SUITES},
               "suites": SOURCE, "splits": counts,
               "built_from": "protoLabsAI/lab experiments/jev",
               "note": "test = scored. calib = fits post-hoc calibration only, never scored. "
                       "train exists upstream for the embed-head arm and is not shipped."},
              open(f"{out}/arms.json", "w"), indent=1)

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    fams = sorted(set(family(s) for s in SUITES))
    print(f"built {out}")
    print(f"  {len(ARMS)} arms · {len(SUITES)} tasks · {len(fams)} families "
          f"({', '.join(fams)}) · {n_vec:,} probability vectors")
    print(f"  {total/1e6:.1f} MB on disk")
    print("\nNOT pushed. Review the card, then publish deliberately.")


if __name__ == "__main__":
    main()
