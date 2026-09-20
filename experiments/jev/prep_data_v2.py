#!/usr/bin/env python3
"""Rung 1 — widen the benchmark from 4 tasks to ~20, grouped into FAMILIES.

Four suites is four leave-one-out folds. That is enough to see the direction of an effect and
not enough to claim cross-task transfer, which is the claim the whole Jev question turns on. This
builds the wider set.

Two things this does that `prep_data.py` did not:

  families   every task declares one. The honest held-out test is leave-one-FAMILY-out: calibrate
             on safety + topic + nli, evaluate on ordinal. Leaving out one task whose three
             siblings share its family leaks the family.
  ordinal    a `score`-shaped family (star ratings) — Jev's third primitive, absent from v0.

The four v0 suites are NOT rebuilt. Their files are left byte-identical so every published v0
number stays valid and v1 is a strict superset.

Readout constraint, enforced here rather than discovered at run time: every option's FIRST token
must be unique within a task, under every tokenizer we decode with. That is what makes
P(first token) == P(option) exact under a grammar admitting only the option strings. A task whose
options collide is reported and skipped, not silently included.

    python prep_data_v2.py --list          # show the roster, build nothing
    python prep_data_v2.py                 # build everything missing
    python prep_data_v2.py --only ag_news emotion
"""
import argparse, json, os, random, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 20260919
N_TEST, N_CALIB, N_TRAIN = 300, 200, 1000
TOKENIZERS = ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-4B"]

V0 = {"injection": "safety", "sentiment": "sentiment", "routing": "topic", "langid": "language"}


# ---------------------------------------------------------------- roster
def pair(a, b):
    return f"{a}\n\n{b}"


TASKS = [
    # ---------------- safety ----------------
    dict(name="offensive", family="safety",
         question="Is this social media post offensive?",
         options=["no", "offensive"],
         load=lambda ld: [(r["text"], ["no", "offensive"][r["label"]])
                          for r in ld("cardiffnlp/tweet_eval", "offensive")["train"]]),
    dict(name="hate", family="safety",
         question="Does this social media post express hate toward a group?",
         options=["no", "hateful"],
         load=lambda ld: [(r["text"], ["no", "hateful"][r["label"]])
                          for r in ld("cardiffnlp/tweet_eval", "hate")["train"]]),
    dict(name="spam", family="safety",
         question="Is this SMS message spam?",
         options=["legitimate", "spam"],
         load=lambda ld: [(r["sms"], ["legitimate", "spam"][r["label"]])
                          for r in ld("ucirvine/sms_spam")["train"]]),

    # ---------------- sentiment / affect ----------------
    dict(name="tweetsent", family="sentiment",
         question="What is the sentiment of this tweet?",
         options=["negative", "neutral", "positive"],
         load=lambda ld: [(r["text"], ["negative", "neutral", "positive"][r["label"]])
                          for r in ld("cardiffnlp/tweet_eval", "sentiment")["train"]]),
    dict(name="emotion", family="sentiment",
         question="Which emotion does this message express?",
         options=["sadness", "joy", "love", "anger", "fear", "surprise"],
         load=lambda ld: [(r["text"], ["sadness", "joy", "love", "anger", "fear", "surprise"][r["label"]])
                          for r in ld("dair-ai/emotion")["train"]]),

    # ---------------- topic / routing ----------------
    dict(name="agnews", family="topic",
         question="Which section of a newspaper does this story belong to?",
         options=["world", "sports", "business", "technology"],
         load=lambda ld: [(r["text"], ["world", "sports", "business", "technology"][r["label"]])
                          for r in ld("fancyzhx/ag_news")["train"]]),
    dict(name="dbpedia", family="topic",
         question="What kind of thing does this encyclopedia entry describe?",
         options=["company", "school", "artist", "athlete", "politician", "transport",
                  "building", "river", "village", "animal", "plant", "album", "film", "book"],
         load=lambda ld: [(r["content"].strip(),
                           ["company", "school", "artist", "athlete", "politician", "transport",
                            "building", "river", "village", "animal", "plant", "album", "film",
                            "book"][r["label"]])
                          for r in ld("fancyzhx/dbpedia_14")["train"]]),
    dict(name="yahoo", family="topic",
         question="Which category does this question belong to?",
         options=["society", "science", "health", "education", "computers", "sports",
                  "business", "entertainment", "family", "politics"],
         load=lambda ld: [((r["question_title"] + " " + r["question_content"]).strip(),
                           ["society", "science", "health", "education", "computers", "sports",
                            "business", "entertainment", "family", "politics"][r["topic"]])
                          for r in ld("community-datasets/yahoo_answers_topics")["train"]]),

    dict(name="finnews", family="sentiment",
         question="What is the market sentiment of this financial news headline?",
         options=["bearish", "bullish", "neutral"],
         load=lambda ld: [(r["text"], ["bearish", "bullish", "neutral"][r["label"]])
                          for r in ld("zeroshot/twitter-financial-news-sentiment")["train"]]),

    dict(name="massive", family="topic",
         question="Which assistant domain does this user utterance belong to?",
         options=["alarm", "audio", "calendar", "cooking", "datetime", "email", "general",
                  "iot", "lists", "music", "news", "play", "qa", "recommendation", "social",
                  "takeaway", "transport", "weather"],
         load=lambda ld: [(r["text"], r["label_text"].split("_")[0])
                          for r in ld("mteb/amazon_massive_intent", "en")["train"]
                          if r["label_text"].split("_")[0] in
                          {"alarm", "audio", "calendar", "cooking", "datetime", "email",
                           "general", "iot", "lists", "music", "news", "play", "qa",
                           "recommendation", "social", "takeaway", "transport", "weather"}]),

    # ---------------- entailment / logic ----------------
    dict(name="rte", family="nli",
         question="Does the first passage entail the second?",
         options=["yes", "no"],
         load=lambda ld: [(pair(r["sentence1"], r["sentence2"]), ["yes", "no"][r["label"]])
                          for r in ld("nyu-mll/glue", "rte")["train"]]),
    dict(name="mrpc", family="nli",
         question="Do these two sentences mean the same thing?",
         options=["different", "equivalent"],
         load=lambda ld: [(pair(r["sentence1"], r["sentence2"]), ["different", "equivalent"][r["label"]])
                          for r in ld("nyu-mll/glue", "mrpc")["train"]]),
    dict(name="qnli", family="nli",
         question="Does the passage contain the answer to the question?",
         options=["yes", "no"],
         load=lambda ld: [(pair(r["question"], r["sentence"]), ["yes", "no"][r["label"]])
                          for r in ld("nyu-mll/glue", "qnli")["train"]]),
    dict(name="snli", family="nli",
         question="Given the first sentence, is the second true?",
         options=["yes", "maybe", "no"],
         load=lambda ld: [(pair(r["premise"], r["hypothesis"]), ["yes", "maybe", "no"][r["label"]])
                          for r in ld("stanfordnlp/snli")["train"] if r["label"] in (0, 1, 2)]),
    dict(name="cola", family="nli",
         question="Is this sentence grammatically acceptable English?",
         options=["unacceptable", "acceptable"],
         load=lambda ld: [(r["sentence"], ["unacceptable", "acceptable"][r["label"]])
                          for r in ld("nyu-mll/glue", "cola")["train"]]),

    # ---------------- ordinal / the `score` shape ----------------
    dict(name="yelpstars", family="ordinal",
         question="How many stars out of 5 does this review give?",
         options=["1", "2", "3", "4", "5"],
         load=lambda ld: [(r["text"], str(r["label"] + 1))
                          for r in ld("Yelp/yelp_review_full")["train"].select(range(40000))]),
    dict(name="sst5", family="ordinal",
         question="On a 1-5 scale from very negative to very positive, how positive is this review?",
         options=["1", "2", "3", "4", "5"],
         load=lambda ld: [(r["text"], str(r["label"] + 1)) for r in ld("SetFit/sst5")["train"]]),
    dict(name="appreviews", family="ordinal",
         question="How many stars out of 5 does this app review give?",
         options=["1", "2", "3", "4", "5"],
         load=lambda ld: [(r["review"], str(r["star"]))
                          for r in ld("sealuzh/app_reviews")["train"]]),
]


# ---------------------------------------------------------------- checks
def first_token_check(options, toks):
    """Every option must start with a distinct token under every tokenizer we decode with."""
    bad = []
    for tname, tok in toks.items():
        ids = [tok.encode(o, add_special_tokens=False)[0] for o in options]
        if len(set(ids)) != len(ids):
            seen = {}
            for o, i in zip(options, ids):
                seen.setdefault(i, []).append(o)
            clash = [v for v in seen.values() if len(v) > 1]
            bad.append(f"{tname}: {clash}")
    return bad


def write_task(name, family, options, question, rows, source):
    rnd = random.Random(f"{SEED}-{name}")
    rows = [r for r in rows if r[0] and r[0].strip()]
    rnd.shuffle(rows)
    n_train = min(N_TRAIN, max(0, len(rows) - N_TEST - N_CALIB))
    parts = {"test": rows[:N_TEST], "calib": rows[N_TEST:N_TEST + N_CALIB],
             "train": rows[N_TEST + N_CALIB:N_TEST + N_CALIB + n_train]}
    os.makedirs(OUT, exist_ok=True)
    for k, v in parts.items():
        with open(f"{OUT}/{name}.{k}.jsonl", "w") as f:
            for i, r in enumerate(v):
                f.write(json.dumps({"id": f"{name}-{k}-{i}", "text": r[0], "label": r[1]},
                                   ensure_ascii=False) + "\n")
    dist = {}
    for _, l in parts["test"]:
        dist[l] = dist.get(l, 0) + 1
    meta = {"suite": name, "family": family, "options": options, "question": question,
            "sizes": {k: len(v) for k, v in parts.items()}, "seed": SEED, "source": source,
            "test_majority": max(dist.values()) / max(1, len(parts["test"]))}
    json.dump(meta, open(f"{OUT}/{name}.meta.json", "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.list:
        fam = {}
        for t in TASKS:
            fam.setdefault(t["family"], []).append(t["name"])
        for f, names in sorted(fam.items()):
            have = [n for n, ff in V0.items() if ff == f]
            print(f"{f:10s} v0: {','.join(have) or '-':12s} new: {', '.join(names)}")
        print(f"\n{len(TASKS)} new + {len(V0)} existing = {len(TASKS)+len(V0)} tasks")
        return

    from transformers import AutoTokenizer
    toks = {}
    for t in TOKENIZERS:
        try:
            toks[t.split("/")[-1]] = AutoTokenizer.from_pretrained(t)
        except Exception as e:
            print(f"!! tokenizer {t}: {e}")
    if not toks:
        sys.exit("no tokenizers available — cannot verify the readout constraint")

    from datasets import load_dataset
    def ld(*args, **kw):
        return load_dataset(*args, **kw)

    todo = [t for t in TASKS if not a.only or t["name"] in a.only]
    ok, skipped = [], []
    for t in todo:
        name = t["name"]
        if os.path.exists(f"{OUT}/{name}.meta.json") and not a.force:
            print(f"{name:12s} exists, skipping (use --force to rebuild)")
            continue
        bad = first_token_check(t["options"], toks)
        if bad:
            print(f"{name:12s} SKIP — first-token collision: {'; '.join(bad)}")
            skipped.append((name, "first-token collision"))
            continue
        try:
            rows = t["load"](ld)
        except Exception as e:
            print(f"{name:12s} SKIP — load failed: {type(e).__name__}: {str(e)[:140]}")
            skipped.append((name, f"load failed: {type(e).__name__}"))
            continue
        if len(rows) < N_TEST + N_CALIB:
            print(f"{name:12s} SKIP — only {len(rows)} rows, need {N_TEST+N_CALIB}")
            skipped.append((name, f"only {len(rows)} rows"))
            continue
        meta = write_task(name, t["family"], t["options"], t["question"], rows,
                          {"note": "see prep_data_v2.py for the exact construction"})
        print(f"{name:12s} {t['family']:10s} k={len(t['options']):2d} "
              f"{meta['sizes']}  majority={meta['test_majority']:.2f}")
        ok.append(name)

    print(f"\nbuilt {len(ok)}, skipped {len(skipped)}")
    for n, why in skipped:
        print(f"  - {n}: {why}")


if __name__ == "__main__":
    main()
