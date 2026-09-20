#!/usr/bin/env python3
"""Reconstruct the item texts for deferral-bench-v0 from their source datasets.

This repo ships item ids, gold labels and model probability vectors — NOT the source texts.
The four sources carry four different licenses, so redistributing them here would be the wrong
call; instead this script rebuilds them deterministically and checks every item against the
SHA-256 in the manifest, so a drifted upstream dataset is caught loudly rather than silently
changing what the benchmark measures.

    pip install datasets pandas pyarrow
    python rebuild.py --out texts/

Writes texts/<suite>.<split>.jsonl with {id, text}. Nothing else in the repo needs them —
the scorer works on the shipped probability vectors alone.
"""
import argparse, hashlib, json, os, random

SEED = 20260919
N_TEST, N_CALIB, N_TRAIN = 300, 200, 1000
LANG = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English",
        "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
        "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
        "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"}


def sources():
    """Exactly the construction used to build the benchmark (prep_data.py)."""
    from datasets import load_dataset, concatenate_datasets
    out = {}

    ds = load_dataset("deepset/prompt-injections")
    d = concatenate_datasets([ds["train"], ds["test"]])
    out["injection"] = [(r["text"], ["benign", "injection"][r["label"]]) for r in d]

    ds = load_dataset("stanfordnlp/sst2")
    out["sentiment"] = [(r["sentence"], ["negative", "positive"][r["label"]])
                        for r in ds["validation"]]

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]
    out["routing"] = [(r["instruction"], r["category"]) for r in ds]

    ds = load_dataset("papluca/language-identification")["test"]
    out["langid"] = [(r["text"], LANG[r["labels"]]) for r in ds]
    return out


def split_rows(rows, name):
    rnd = random.Random(f"{SEED}-{name}")
    rows = list(rows)
    rnd.shuffle(rows)
    n_train = min(N_TRAIN, max(0, len(rows) - N_TEST - N_CALIB))
    return {"test": rows[:N_TEST],
            "calib": rows[N_TEST:N_TEST + N_CALIB],
            "train": rows[N_TEST + N_CALIB:N_TEST + N_CALIB + n_train]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="texts")
    ap.add_argument("--splits", default="test,calib,train")
    a = ap.parse_args()
    import pandas as pd

    os.makedirs(a.out, exist_ok=True)
    src = sources()
    bad = total = 0
    for suite, rows in src.items():
        parts = split_rows(rows, suite)
        for split in a.splits.split(","):
            man_path = f"manifest/{suite}.{split}.parquet"
            if not os.path.exists(man_path):
                continue
            man = pd.read_parquet(man_path)
            items = parts[split]
            if len(items) != len(man):
                print(f"!! {suite}.{split}: rebuilt {len(items)} rows, manifest has {len(man)} "
                      f"— the upstream dataset has changed size")
                bad += abs(len(items) - len(man))
                continue
            recs, mism = [], 0
            for (text, label), (_, m) in zip(items, man.iterrows()):
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h != m["text_sha256"]:
                    mism += 1
                recs.append({"id": m["id"], "text": text})
            total += len(recs)
            bad += mism
            with open(f"{a.out}/{suite}.{split}.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            flag = "OK" if mism == 0 else f"{mism} HASH MISMATCHES"
            print(f"{suite:10s} {split:6s} n={len(recs):4d}  {flag}")
    print(f"\n{total} items rebuilt, {bad} mismatched.")
    if bad:
        print("A mismatch means the upstream dataset changed. The shipped probability vectors\n"
              "still correspond to the ORIGINAL texts — do not mix them with rebuilt items that\n"
              "failed their hash.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
