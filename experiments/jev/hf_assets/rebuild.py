#!/usr/bin/env python3
"""Reconstruct the item texts for deferral-bench-v1 from their source datasets.

This repo ships item ids, gold labels and model probability vectors — NOT the source texts.
Twenty-two tasks drawn from twenty different upstream datasets carry twenty different licenses,
so redistributing them here would be the wrong call. This script rebuilds them deterministically
and checks every item against the SHA-256 in the manifest, so a drifted upstream dataset is
caught loudly rather than silently changing what the benchmark measures.

    pip install datasets pandas pyarrow
    python rebuild.py --out texts/

Writes texts/<task>.<split>.jsonl with {id, text}. Nothing else in the repo needs them — the
scorer works on the shipped probability vectors alone.
"""
import argparse, hashlib, json, os, random

SEED = 20260919
N_TEST, N_CALIB, N_TRAIN = 300, 200, 1000

LANG = {"ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English",
        "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
        "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
        "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"}
MASSIVE = {"alarm", "audio", "calendar", "cooking", "datetime", "email", "general", "iot",
           "lists", "music", "news", "play", "qa", "recommendation", "social", "takeaway",
           "transport", "weather"}
DBP = ["company", "school", "artist", "athlete", "politician", "transport", "building", "river",
       "village", "animal", "plant", "album", "film", "book"]
YAHOO = ["society", "science", "health", "education", "computers", "sports", "business",
         "entertainment", "family", "politics"]
EMO = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def pair(a, b):
    return f"{a}\n\n{b}"


def build(ld, cat):
    """Exactly the constructions used to build the benchmark (prep_data.py + prep_data_v2.py)."""
    T = {}

    # ---- v0 four ----
    d = cat([ld("deepset/prompt-injections")["train"], ld("deepset/prompt-injections")["test"]])
    T["injection"] = [(r["text"], ["benign", "injection"][r["label"]]) for r in d]
    T["sentiment"] = [(r["sentence"], ["negative", "positive"][r["label"]])
                      for r in ld("stanfordnlp/sst2")["validation"]]
    T["routing"] = [(r["instruction"], r["category"]) for r in
                    ld("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]]
    T["langid"] = [(r["text"], LANG[r["labels"]])
                   for r in ld("papluca/language-identification")["test"]]

    # ---- safety ----
    T["offensive"] = [(r["text"], ["no", "offensive"][r["label"]])
                      for r in ld("cardiffnlp/tweet_eval", "offensive")["train"]]
    T["hate"] = [(r["text"], ["no", "hateful"][r["label"]])
                 for r in ld("cardiffnlp/tweet_eval", "hate")["train"]]
    T["spam"] = [(r["sms"], ["legitimate", "spam"][r["label"]])
                 for r in ld("ucirvine/sms_spam")["train"]]

    # ---- sentiment ----
    T["tweetsent"] = [(r["text"], ["negative", "neutral", "positive"][r["label"]])
                      for r in ld("cardiffnlp/tweet_eval", "sentiment")["train"]]
    T["emotion"] = [(r["text"], EMO[r["label"]]) for r in ld("dair-ai/emotion")["train"]]
    T["finnews"] = [(r["text"], ["bearish", "bullish", "neutral"][r["label"]])
                    for r in ld("zeroshot/twitter-financial-news-sentiment")["train"]]

    # ---- topic ----
    T["agnews"] = [(r["text"], ["world", "sports", "business", "technology"][r["label"]])
                   for r in ld("fancyzhx/ag_news")["train"]]
    T["dbpedia"] = [(r["content"].strip(), DBP[r["label"]])
                    for r in ld("fancyzhx/dbpedia_14")["train"]]
    T["yahoo"] = [((r["question_title"] + " " + r["question_content"]).strip(), YAHOO[r["topic"]])
                  for r in ld("community-datasets/yahoo_answers_topics")["train"]]
    T["massive"] = [(r["text"], r["label_text"].split("_")[0])
                    for r in ld("mteb/amazon_massive_intent", "en")["train"]
                    if r["label_text"].split("_")[0] in MASSIVE]

    # ---- nli ----
    T["rte"] = [(pair(r["sentence1"], r["sentence2"]), ["yes", "no"][r["label"]])
                for r in ld("nyu-mll/glue", "rte")["train"]]
    T["mrpc"] = [(pair(r["sentence1"], r["sentence2"]), ["different", "equivalent"][r["label"]])
                 for r in ld("nyu-mll/glue", "mrpc")["train"]]
    T["qnli"] = [(pair(r["question"], r["sentence"]), ["yes", "no"][r["label"]])
                 for r in ld("nyu-mll/glue", "qnli")["train"]]
    T["snli"] = [(pair(r["premise"], r["hypothesis"]), ["yes", "maybe", "no"][r["label"]])
                 for r in ld("stanfordnlp/snli")["train"] if r["label"] in (0, 1, 2)]
    T["cola"] = [(r["sentence"], ["unacceptable", "acceptable"][r["label"]])
                 for r in ld("nyu-mll/glue", "cola")["train"]]

    # ---- ordinal ----
    T["yelpstars"] = [(r["text"], str(r["label"] + 1))
                      for r in ld("Yelp/yelp_review_full")["train"].select(range(40000))]
    T["sst5"] = [(r["text"], str(r["label"] + 1)) for r in ld("SetFit/sst5")["train"]]
    T["appreviews"] = [(r["review"], str(r["star"])) for r in ld("sealuzh/app_reviews")["train"]]
    return T


def split_rows(rows, name):
    rnd = random.Random(f"{SEED}-{name}")
    rows = [r for r in rows if r[0] and r[0].strip()] if name not in (
        "injection", "sentiment", "routing", "langid") else list(rows)
    rnd.shuffle(rows)
    n_train = min(N_TRAIN, max(0, len(rows) - N_TEST - N_CALIB))
    return {"test": rows[:N_TEST], "calib": rows[N_TEST:N_TEST + N_CALIB],
            "train": rows[N_TEST + N_CALIB:N_TEST + N_CALIB + n_train]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="texts")
    ap.add_argument("--splits", default="test,calib")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    from datasets import load_dataset, concatenate_datasets

    os.makedirs(a.out, exist_ok=True)
    src = build(load_dataset, concatenate_datasets)
    bad = total = 0
    for task, rows in sorted(src.items()):
        if a.tasks and task not in a.tasks:
            continue
        parts = split_rows(rows, task)
        for split in a.splits.split(","):
            man_path = f"manifest/{task}.{split}.parquet"
            if not os.path.exists(man_path):
                continue
            man = pd.read_parquet(man_path)
            items = parts[split]
            if len(items) != len(man):
                print(f"!! {task}.{split}: rebuilt {len(items)} rows, manifest has {len(man)}")
                bad += abs(len(items) - len(man))
                continue
            recs, mism = [], 0
            for (text, _label), (_, m) in zip(items, man.iterrows()):
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != m["text_sha256"]:
                    mism += 1
                recs.append({"id": m["id"], "text": text})
            total += len(recs)
            bad += mism
            with open(f"{a.out}/{task}.{split}.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not a.quiet:
                print(f"{task:12s} {split:6s} n={len(recs):4d}  "
                      f"{'OK' if mism == 0 else str(mism) + ' HASH MISMATCHES'}")
    print(f"\n{total} items rebuilt, {bad} mismatched.")
    if bad:
        print("A mismatch means the upstream dataset changed. The shipped probability vectors\n"
              "still correspond to the ORIGINAL texts — do not mix them with rebuilt items that\n"
              "failed their hash.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
