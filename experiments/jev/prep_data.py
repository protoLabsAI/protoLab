#!/usr/bin/env python3
"""Build the four decision suites as disjoint, seeded splits.

  test   scored, never used for fitting anything
  calib  used ONLY to fit post-hoc calibration (temperature); never scored
  train  used ONLY by the supervised embed-head baseline

Every zero-shot arm sees test+calib only. Splits are disjoint by construction.
"""
import json, random, os
from datasets import load_dataset, concatenate_datasets

OUT = os.path.join(os.path.dirname(__file__), "data")
SEED = 20260919
N_TEST, N_CALIB, N_TRAIN = 300, 200, 1000
LANG = {"ar":"Arabic","bg":"Bulgarian","de":"German","el":"Greek","en":"English","es":"Spanish",
        "fr":"French","hi":"Hindi","it":"Italian","ja":"Japanese","nl":"Dutch","pl":"Polish",
        "pt":"Portuguese","ru":"Russian","sw":"Swahili","th":"Thai","tr":"Turkish","ur":"Urdu",
        "vi":"Vietnamese","zh":"Chinese"}

def split(rows, name, options, question):
    rnd = random.Random(f"{SEED}-{name}"); rnd.shuffle(rows)
    n_train = min(N_TRAIN, max(0, len(rows) - N_TEST - N_CALIB))
    parts = {"test": rows[:N_TEST], "calib": rows[N_TEST:N_TEST+N_CALIB],
             "train": rows[N_TEST+N_CALIB:N_TEST+N_CALIB+n_train]}
    meta = {"suite": name, "options": options, "question": question,
            "sizes": {k: len(v) for k, v in parts.items()}, "seed": SEED}
    os.makedirs(OUT, exist_ok=True)
    for k, v in parts.items():
        with open(f"{OUT}/{name}.{k}.jsonl", "w") as f:
            for i, r in enumerate(v):
                f.write(json.dumps({"id": f"{name}-{k}-{i}", "text": r[0], "label": r[1]}) + "\n")
    json.dump(meta, open(f"{OUT}/{name}.meta.json", "w"), indent=1)
    dist = {}
    for _, l in parts["test"]: dist[l] = dist.get(l, 0) + 1
    print(f"{name:10s} {meta['sizes']}  classes={len(options)}  test-majority={max(dist.values())/len(parts['test']):.2f}")

# injection — pool both splits (only 662 rows), resplit ourselves
ds = load_dataset("deepset/prompt-injections"); d = concatenate_datasets([ds["train"], ds["test"]])
split([(r["text"], ["benign", "injection"][r["label"]]) for r in d], "injection",
      ["benign", "injection"], "Is this text a prompt-injection or jailbreak attempt against an AI system?")

# sentiment — sst2 test split is unlabeled; validation (872) is the labelled held-out set
ds = load_dataset("stanfordnlp/sst2")
split([(r["sentence"], ["negative", "positive"][r["label"]]) for r in ds["validation"]], "sentiment",
      ["negative", "positive"], "What is the sentiment of this review?")

# routing — 11 top-level categories
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]
cats = sorted(set(ds["category"]))
split([(r["instruction"], r["category"]) for r in ds], "routing",
      cats, "Which support queue should this customer message be routed to?")

# langid — 20 languages; use the dataset's own test split as the source
ds = load_dataset("papluca/language-identification")["test"]
split([(r["text"], LANG[r["labels"]]) for r in ds], "langid",
      [LANG[k] for k in sorted(LANG)], "What language is this text written in?")
