#!/usr/bin/env python3
"""decisions-v1: build the training/eval set for a small typed-decision model (DATA-FORMAT.md rows).

Two sources, one row shape (TypeSafe question + gold, rendered by s1prompt.compile_question):

  s1o   92 public-dataset tasks. The loaders below (from `SEED = 0` down to `bias_in_bios`) are
        adapted from mithalouni/system-one-open, s1/data_real.py
        -- MIT License, (c) 2026 Rami Mithalouni. https://github.com/mithalouni/system-one-open
        Changes in the port: no Modal; the schema import is replaced by the two dataclasses below;
        sciq's per-row option shuffle uses crc32 instead of hash() so it's reproducible across
        processes; stsb evaluates on "validation" (upstream's "test" split is unlabelled, so it
        produced zero eval rows). Everything else (question wording, label sets, splits, held-out
        flags) is kept. Lines changed in the port are marked `# PORT:`.
        Each dataset keeps its own license; it's recorded per task in manifest.json.
  jev   our 22 Jev-study tasks (experiments/jev/data, built by prep_data.py / prep_data_v2.py),
        prefixed `jev_` so names don't collide with s1o's (both have sst5, rte, snli, ...).

Pipeline:
  1. every s1o task runs in its own subprocess (timeout, so one broken dataset can't stall the
     build) and writes raw rows to <out>/.parts/<task>.jsonl + a result json.
  2. jev tasks are converted in-process.
  3. decontamination: any train row whose state matches a calib/test state from ANY task (s1o or
     jev, held-out or not) is dropped. The jev tasks were drawn from the same HF splits that
     several s1o tasks train on (tweet_eval, sms_spam, SetFit/sst5, glue, ...), so without this
     the jev eval texts would sit in the training set under different question wording.
  4. per-split caps (deterministic), ids, <task>.jsonl + manifest.json.

Run (CPU only, never the GPUs):
  CUDA_VISIBLE_DEVICES= ~/dev/quant-env/bin/python build_dataset.py --jobs 8
  ~/dev/quant-env/bin/python build_dataset.py --validate-only
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")          # token lives here
# Raw dataset downloads must NOT land under /mnt/models (models-only drive). Forced, not
# setdefault: ~/.bashrc exports HF_HUB_CACHE=/mnt/models/huggingface/hub, which would win.
os.environ["HF_DATASETS_CACHE"] = "/mnt/data/datasets/hf-cache"
os.environ["HF_HUB_CACHE"] = "/mnt/data/datasets/hf-cache/hub"
os.environ["HF_XET_CACHE"] = "/mnt/data/datasets/hf-cache/xet"
os.environ["HF_ASSETS_CACHE"] = "/mnt/data/datasets/hf-cache/assets"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse, ast, hashlib, json, random, re, subprocess, sys, time, types, zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
JEV_DATA = os.path.join(HERE, "..", "jev", "data")
DEFAULT_OUT = "/mnt/data/datasets/decisions-v1"
MAX_CHOICE, MAX_SCORE = 20, 10          # s1serve.py serving limits


# ── stand-ins for system-one-open's s1/schema.py types (only what the loaders use) ──────────────
NONE_OPT = "none of the above"


@dataclass
class Q:
    text: str
    options: list
    gold: int = -1
    kind: str = "choice"            # choice | noul | score
    descs: list | None = None
    id: str | None = None


@dataclass
class Example:
    state: str
    qs: list
    task: str = ""
    meta: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# BEGIN adapted from mithalouni/system-one-open s1/data_real.py (MIT, (c) 2026 Rami Mithalouni)
# ════════════════════════════════════════════════════════════════════════════════════════════════
SEED = 0
TRAIN_CAP = 12000
EVAL_CAP = 600
TASKS = {}


def task(name, heldout=False, group="misc"):
    def deco(fn):
        TASKS[name] = dict(loader=fn, heldout=heldout, group=group)
        return fn
    return deco


def _nice(label):
    return str(label).replace("_", " ").strip()


def _clip(s, n):
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[:n] + " ..."


def _ld(name, cfg=None, split=None, rev=None):
    from datasets import load_dataset
    kw = dict(split=split)
    if rev:
        kw["revision"] = rev
    return load_dataset(name, cfg, **kw) if cfg else load_dataset(name, **kw)


def _sub(ds, n, seed=SEED):
    if n is None or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _names(ds, col="label"):
    return [_nice(n) for n in ds.features[col].names]


def _cls(ds, text_fn, label_fn, question, names, task, cap, kind="choice", descs=None):
    ds = _sub(ds, cap)
    out = []
    for r in ds:
        try:
            y = label_fn(r)
        except Exception:
            continue
        if y is None or y < 0 or y >= len(names):
            continue
        ctx = text_fn(r)
        if not ctx or not str(ctx).strip():
            continue
        out.append(Example(str(ctx).strip(), [Q(question, list(names), int(y), kind=kind, descs=descs)], task))
    return out


def _pair(name, cfg, tr_split, ev_split, rev=None):
    return _ld(name, cfg, tr_split, rev), _ld(name, cfg, ev_split, rev)


def _both(fn, tr, ev, t):
    return fn(tr, t, TRAIN_CAP), fn(ev, t, EVAL_CAP)


def _yesno(question, ctx_fn, label_fn, ds, t, cap, descs=None):
    return _cls(ds, ctx_fn, label_fn, question, ["no", "yes"], t, cap, kind="noul", descs=descs)


def _scale_q(text, legend):
    keys = sorted(legend, key=lambda k: float(k))
    return keys, Q(text, [str(k) for k in keys], -1, kind="score", descs=[legend[k] for k in keys])


def _score(ds, ctx_fn, val_fn, text, legend, t, cap):
    keys, proto = _scale_q(text, legend)
    ds = _sub(ds, cap); out = []
    for r in ds:
        try:
            v = val_fn(r)
        except Exception:
            continue
        if v is None or v not in keys:
            continue
        ctx = ctx_fn(r)
        if not ctx or not str(ctx).strip():
            continue
        out.append(Example(str(ctx).strip(), [Q(proto.text, proto.options, keys.index(v), kind="score", descs=proto.descs)], t))
    return out


# =============================================================== intents / routing
@task("clinc_oos", group="intent")
def _clinc():
    tr, ev = _pair("clinc/clinc_oos", "plus", "train", "test")
    names = _names(tr, "intent"); oos = names.index("oos"); names[oos] = "none of the above (out of scope)"
    q = "What is the intent of this user message?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["intent"], q, names, t, cap)
    return _both(f, tr, ev, "clinc_oos")


@task("banking77", group="intent")
def _banking():
    tr, ev = _pair("mteb/banking77", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    q = "Which banking-support intent does this customer message express?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], q, [_nice(n) for n in names], t, cap)
    return _both(f, tr, ev, "banking77")


@task("massive_intent", group="intent")
def _massive():
    tr, ev = _pair("SetFit/amazon_massive_intent_en-US", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    q = "What is the intent of this voice-assistant utterance?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], q, [_nice(n) for n in names], t, cap)
    return _both(f, tr, ev, "massive_intent")


@task("massive_scenario", heldout=True, group="intent")
def _massive_sc():
    tr, ev = _pair("SetFit/amazon_massive_scenario_en-US", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    q = "Which scenario (domain) does this voice-assistant utterance belong to?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], q, [_nice(n) for n in names], t, cap)
    return _both(f, tr, ev, "massive_scenario")


@task("bitext_support", group="intent")
def _bitext():
    ds = _ld("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "train").shuffle(seed=SEED)
    cats = sorted(set(ds["category"])); intents = sorted(set(ds["intent"]))
    def conv(d, t, cap):
        out = []
        for r in _sub(d, cap):
            out.append(Example(r["instruction"].strip(), [
                Q("Which support category does this customer request belong to?", [_nice(c) for c in cats], cats.index(r["category"])),
                Q("What is the specific intent of the request?", [_nice(i) for i in intents], intents.index(r["intent"]))], t))
        return out
    n_ev = 1500
    return conv(ds.select(range(n_ev, len(ds))), "bitext_support", TRAIN_CAP), conv(ds.select(range(n_ev)), "bitext_support", EVAL_CAP)


@task("support_tickets", group="intent")
def _tickets():
    ds = _ld("Tobi-Bueck/customer-support-tickets", None, "train").shuffle(seed=SEED)
    ds = ds.filter(lambda r: (r.get("language") or "en") == "en" and r.get("body"))
    queues = sorted(set(x for x in ds["queue"] if x)); prios = ["low", "medium", "high"]
    def conv(d, t, cap):
        out = []
        for r in _sub(d, cap):
            if not r.get("queue") or r["queue"] not in queues:
                continue
            ctx = f"Subject: {r.get('subject') or ''}\n\n{_clip(r['body'], 2500)}"
            qs = [Q("Which team queue should this ticket be routed to?", queues, queues.index(r["queue"]))]
            p = (r.get("priority") or "").lower()
            if p in prios:
                qs.append(Q("What priority should this ticket get?", prios, prios.index(p), kind="score",
                            descs=["can wait; no business impact", "should be handled soon; some impact", "urgent; blocking or severe impact"]))
            out.append(Example(ctx, qs, t))
        return out
    n_ev = 1000
    return conv(ds.select(range(n_ev, len(ds))), "support_tickets", TRAIN_CAP), conv(ds.select(range(n_ev)), "support_tickets", EVAL_CAP)


# =============================================================== topic
@task("ag_news", group="topic")
def _ag():
    tr, ev = _pair("fancyzhx/ag_news", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Which topic does this news article belong to?", ["world", "sports", "business", "science and technology"], t, cap)
    return _both(f, tr, ev, "ag_news")


@task("dbpedia", group="topic")
def _dbp():
    tr, ev = _pair("fancyzhx/dbpedia_14", None, "train", "test")
    names = _names(tr)
    f = lambda ds, t, cap: _cls(ds, lambda r: f"{r['title']}: {r['content']}", lambda r: r["label"], "What kind of entity does this text describe?", names, t, cap)
    return _both(f, tr, ev, "dbpedia")


@task("yahoo_topics", group="topic")
def _yahoo():
    tr, ev = _pair("community-datasets/yahoo_answers_topics", None, "train", "test")
    names = _names(tr, "topic")
    f = lambda ds, t, cap: _cls(ds, lambda r: f"{r['question_title']}\n{_clip(r['question_content'], 800)}", lambda r: r["topic"], "Which topic category does this question belong to?", names, t, cap)
    return _both(f, tr, ev, "yahoo_topics")


@task("newsgroups", group="topic")
def _ng():
    tr, ev = _pair("SetFit/20_newsgroups", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    tops = sorted(set(n.split(".")[0] for n in names))
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            if not r["text"].strip():
                continue
            lt = r["label_text"]
            out.append(Example(_clip(r["text"], 2500), [
                Q("Which top-level newsgroup hierarchy does this post belong to?", tops, tops.index(lt.split(".")[0])),
                Q("Which newsgroup was this post published in?", names, idx[lt])], t))
        return out
    return _both(conv, tr, ev, "newsgroups")


@task("bbc_news", heldout=True, group="topic")
def _bbc():
    tr, ev = _pair("SetFit/bbc-news", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: _clip(r["text"], 2500), lambda r: idx[r["label_text"]], "Which section does this BBC news article belong to?", names, t, cap)
    return _both(f, tr, ev, "bbc_news")


@task("trec", heldout=True, group="topic")
def _trec():
    tr, ev = _pair("SetFit/TREC-QC", None, "train", "test")
    names = sorted(set(tr["label_coarse_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_coarse_text"]], "What kind of answer is this question asking for?", names, t, cap)
    return _both(f, tr, ev, "trec")


@task("student_questions", heldout=True, group="topic")
def _stuq():
    tr, ev = _pair("SetFit/student-question-categories", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], "Which subject is this student question about?", names, t, cap)
    return _both(f, tr, ev, "student_questions")


@task("dolly_category", heldout=True, group="topic")
def _dolly():
    ds = _ld("argilla/databricks-dolly-15k-curated-en", None, "train").shuffle(seed=SEED)
    names = sorted(set(ds["category"]))
    f = lambda d, t, cap: _cls(d, lambda r: f"Instruction: {r['new-instruction']['value'][0] if r['new-instruction']['value'] else r['original-instruction']}\nContext: {_clip(r['original-context'], 800)}", lambda r: names.index(r["category"]), "What type of task is this instruction?", [_nice(n) for n in names], t, cap)
    return [], f(ds, "dolly_category", EVAL_CAP)


# =============================================================== sentiment / emotion / scales
@task("imdb", group="sentiment")
def _imdb():
    tr, ev = _pair("stanfordnlp/imdb", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: _clip(r["text"], 2500), lambda r: r["label"], "What is the sentiment of this movie review?", ["negative", "positive"], t, cap)
    return _both(f, tr, ev, "imdb")


@task("sst2", group="sentiment")
def _sst2():
    tr, ev = _pair("stanfordnlp/sst2", None, "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["sentence"], lambda r: r["label"], "What is the sentiment of this sentence?", ["negative", "positive"], t, cap)
    return _both(f, tr, ev, "sst2")


@task("sst5", group="sentiment")
def _sst5():
    tr, ev = _pair("SetFit/sst5", None, "train", "test")
    legend = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
    f = lambda ds, t, cap: _score(ds, lambda r: r["text"], lambda r: int(r["label"]), "How positive is the sentiment of this sentence?", legend, t, cap)
    return _both(f, tr, ev, "sst5")


@task("yelp", group="sentiment")
def _yelp():
    tr, ev = _pair("Yelp/yelp_review_full", None, "train", "test")
    legend = {1: "1 star - terrible", 2: "2 stars - poor", 3: "3 stars - average", 4: "4 stars - good", 5: "5 stars - excellent"}
    f = lambda ds, t, cap: _score(ds, lambda r: _clip(r["text"], 2500), lambda r: int(r["label"]) + 1, "How many stars did this reviewer give?", legend, t, cap)
    return _both(f, tr, ev, "yelp")


@task("amazon_stars", group="sentiment")
def _amz():
    tr, ev = _pair("SetFit/amazon_reviews_multi_en", None, "train", "test")
    legend = {1: "1 star", 2: "2 stars", 3: "3 stars", 4: "4 stars", 5: "5 stars"}
    f = lambda ds, t, cap: _score(ds, lambda r: _clip(r["text"], 2000), lambda r: int(r["label"]) + 1, "What star rating accompanies this product review?", legend, t, cap)
    return _both(f, tr, ev, "amazon_stars")


@task("emotion", group="sentiment")
def _emo():
    tr, ev = _pair("dair-ai/emotion", "split", "train", "test")
    names = _names(tr)
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Which emotion does this text express?", names, t, cap)
    return _both(f, tr, ev, "emotion")


@task("go_emotions", group="sentiment")
def _goemo():
    tr, ev = _pair("google-research-datasets/go_emotions", "simplified", "train", "test")
    names = [_nice(n) for n in tr.features["labels"].feature.names]
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            if len(r["labels"]) != 1:
                continue
            out.append(Example(r["text"], [Q("Which emotion is most strongly expressed in this comment?", names, int(r["labels"][0]))], t))
        return out
    return _both(conv, tr, ev, "go_emotions")


@task("tweet_sentiment", group="sentiment")
def _tws():
    tr, ev = _pair("cardiffnlp/tweet_eval", "sentiment", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the sentiment of this tweet?", ["negative", "neutral", "positive"], t, cap)
    return _both(f, tr, ev, "tweet_sentiment")


@task("tweet_emotion", group="sentiment")
def _twe():
    tr, ev = _pair("cardiffnlp/tweet_eval", "emotion", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Which emotion does this tweet express?", ["anger", "joy", "optimism", "sadness"], t, cap)
    return _both(f, tr, ev, "tweet_emotion")


@task("tweet_irony", heldout=True, group="sentiment")
def _twi():
    tr, ev = _pair("cardiffnlp/tweet_eval", "irony", "train", "test")
    f = lambda ds, t, cap: _yesno("Is this tweet ironic?", lambda r: r["text"], lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "tweet_irony")


@task("fin_sentiment", heldout=True, group="sentiment")
def _fin():
    tr, ev = _pair("zeroshot/twitter-financial-news-sentiment", None, "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the sentiment of this financial news tweet?", ["bearish", "bullish", "neutral"], t, cap)
    return _both(f, tr, ev, "fin_sentiment")


@task("fin_phrasebank", heldout=True, group="sentiment")
def _fpb():
    ds = _ld("AdaptLLM/finance-tasks", "FPB", "test")
    def conv(d, t, cap):
        out = []
        for r in _sub(d, cap):
            opts = [o.strip() for o in r["options"]]
            out.append(Example(_clip(r["input"], 2000), [Q("What is the sentiment of this financial news sentence?", opts, int(r["gold_index"]))], t))
        return out
    return [], conv(ds, "fin_phrasebank", EVAL_CAP)


@task("cr_reviews", heldout=True, group="sentiment")
def _cr():
    tr, ev = _pair("SetFit/SentEval-CR", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the sentiment of this product review sentence?", ["negative", "positive"], t, cap)
    return _both(f, tr, ev, "cr_reviews")


@task("counterfactual", group="sentiment")
def _cf():
    tr, ev = _pair("SetFit/amazon_counterfactual_en", None, "train", "test")
    f = lambda ds, t, cap: _yesno("Does this review sentence describe a counterfactual (something that did not happen)?", lambda r: r["text"], lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "counterfactual")


@task("subjectivity", group="sentiment")
def _subj():
    tr, ev = _pair("SetFit/subj", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Is this sentence objective or subjective?", ["objective", "subjective"], t, cap)
    return _both(f, tr, ev, "subjectivity")


@task("stsb", group="scale")
def _stsb():
    tr, ev = _pair("SetFit/stsb", None, "train", "validation")  # PORT: upstream used "test", which is unlabelled (-1) -> 0 eval rows
    legend = {0: "completely unrelated", 1: "on the same topic but different meaning", 2: "share some details", 3: "roughly equivalent, important details differ", 4: "mostly equivalent, minor details differ", 5: "completely equivalent in meaning"}
    f = lambda ds, t, cap: _score(ds, lambda r: f"Text 1: {r['text1']}\nText 2: {r['text2']}", lambda r: int(round(float(r["label"]))), "How similar in meaning are the two texts?", legend, t, cap)
    return _both(f, tr, ev, "stsb")


@task("helpsteer2", group="scale")
def _hs2():
    tr, ev = _pair("nvidia/HelpSteer2", None, "train", "validation")
    legends = {
        "helpfulness": {0: "not helpful", 1: "slightly helpful", 2: "partially helpful", 3: "mostly helpful", 4: "extremely helpful"},
        "correctness": {0: "completely incorrect", 1: "mostly incorrect", 2: "partially correct", 3: "mostly correct", 4: "completely correct"},
        "coherence": {0: "incoherent", 1: "mostly incoherent", 2: "partially coherent", 3: "mostly coherent", 4: "perfectly coherent"},
        "complexity": {0: "basic, anyone could write it", 1: "simple", 2: "intermediate", 3: "advanced", 4: "expert-level"},
        "verbosity": {0: "far too short", 1: "somewhat short", 2: "adequate length", 3: "somewhat long", 4: "far too long"}}
    qtext = {"helpfulness": "How helpful is the response to the prompt?", "correctness": "How correct and accurate is the response?",
             "coherence": "How coherent and clear is the response?", "complexity": "What level of expertise does the response's content require?",
             "verbosity": "How verbose is the response relative to what the prompt asks for?"}
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            ctx = json.dumps({"prompt": _clip(r["prompt"], 2500), "response": _clip(r["response"], 2500)}, ensure_ascii=False, indent=1)
            qs = []
            for a, leg in legends.items():
                v = int(r[a])
                if 0 <= v <= 4:
                    keys, proto = _scale_q(qtext[a], leg); qs.append(Q(proto.text, proto.options, keys.index(v), kind="score", descs=proto.descs))
            out.append(Example(ctx, qs, t))
        return out
    return _both(conv, tr, ev, "helpsteer2")


@task("hate_speech_scales", group="scale")
def _mhs():
    ds = _ld("ucberkeley-dlab/measuring-hate-speech", None, "train")
    cols = ["hatespeech", "insult", "dehumanize", "violence", "sentiment"]
    agg = defaultdict(lambda: defaultdict(list)); text = {}
    for r in ds:
        cid = r["comment_id"]; text[cid] = r["text"]
        for c in cols:
            if r[c] is not None:
                agg[cid][c].append(float(r[c]))
    L5 = {0: "not at all", 1: "slightly", 2: "somewhat", 3: "very", 4: "extremely"}
    hs_leg = {0: "not hate speech", 1: "unclear or borderline", 2: "hate speech"}
    sent_leg = {0: "strongly positive", 1: "positive", 2: "neutral", 3: "negative", 4: "strongly negative"}
    ids = sorted(agg); random.Random(SEED).shuffle(ids)
    def conv(idl, t, cap):
        out = []
        for cid in idl[:cap]:
            a = agg[cid]
            if not all(a[c] for c in cols):
                continue
            m = {c: int(round(sum(a[c]) / len(a[c]))) for c in cols}
            qs = []
            k, p = _scale_q("Is this comment hate speech?", hs_leg); qs.append(Q(p.text, p.options, k.index(min(2, max(0, m["hatespeech"]))), kind="score", descs=p.descs))
            for c, qt in [("insult", "How insulting is this comment?"), ("dehumanize", "How dehumanizing is this comment?"), ("violence", "How much does this comment incite violence?")]:
                k, p = _scale_q(qt, L5); qs.append(Q(p.text, p.options, k.index(min(4, max(0, m[c]))), kind="score", descs=p.descs))
            k, p = _scale_q("What is the sentiment of this comment?", sent_leg); qs.append(Q(p.text, p.options, k.index(min(4, max(0, m["sentiment"]))), kind="score", descs=p.descs))
            out.append(Example(_clip(text[cid], 2000), qs, t))
        return out
    return conv(ids[1500:], "hate_speech_scales", TRAIN_CAP), conv(ids[:1500], "hate_speech_scales", EVAL_CAP)


@task("liar2", group="scale")
def _liar():
    tr, ev = _pair("chengxuphd/liar2", None, "train", "test")
    legend = {0: "pants on fire (absurdly false)", 1: "false", 2: "barely true", 3: "half true", 4: "mostly true", 5: "true"}
    f = lambda ds, t, cap: _score(ds, lambda r: f"Statement: {r['statement']}\nSpeaker: {r['speaker']} ({_clip(r['speaker_description'] or '', 300)})\nSubject: {r['subject']}\nDate: {r['date']}",
                                  lambda r: int(r["label"]), "How truthful is this political statement, as rated by fact-checkers?", legend, t, cap)
    return _both(f, tr, ev, "liar2")


@task("prosocial_safety", group="scale")
def _pro():
    tr, ev = _pair("allenai/prosocial-dialog", None, "train", "validation")
    legend = {0: "casual, no safety concern", 1: "possibly needs caution", 2: "probably needs caution", 3: "needs caution", 4: "needs intervention"}
    lab = {"__casual__": 0, "__possibly_needs_caution__": 1, "__probably_needs_caution__": 2, "__needs_caution__": 3, "__needs_intervention__": 4}
    f = lambda ds, t, cap: _score(ds, lambda r: r["context"], lambda r: lab.get(r["safety_label"]), "How much does this utterance call for a cautious or corrective response?", legend, t, cap)
    return _both(f, tr, ev, "prosocial_safety")


# =============================================================== moderation / safety / spam
@task("tweet_offensive", group="moderation")
def _two():
    tr, ev = _pair("cardiffnlp/tweet_eval", "offensive", "train", "test")
    f = lambda ds, t, cap: _yesno("Is this tweet offensive?", lambda r: r["text"], lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "tweet_offensive")


@task("tweet_hate", group="moderation")
def _twh():
    tr, ev = _pair("cardiffnlp/tweet_eval", "hate", "train", "test")
    f = lambda ds, t, cap: _yesno("Does this tweet contain hate speech?", lambda r: r["text"], lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "tweet_hate")


@task("hate_offensive", group="moderation")
def _hso():
    tr, ev = _pair("SetFit/hate_speech_offensive", None, "train", "test")
    names = ["hate speech", "offensive language", "neither"]
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "How should this tweet be classified for moderation?", names, t, cap)
    return _both(f, tr, ev, "hate_offensive")


@task("civil_comments", group="moderation")
def _civil():
    tr = _ld("google/civil_comments", None, "train[:300000]"); ev = _ld("google/civil_comments", None, "test[:40000]")
    def conv(ds, t, cap):
        ds = ds.shuffle(seed=SEED); out = []
        pos = neg = 0
        for r in ds:
            y = int(r["toxicity"] >= 0.5)
            if (y and pos >= cap // 2) or (not y and neg >= cap // 2):
                continue
            pos += y; neg += (1 - y)
            qs = [Q("Is this comment toxic?", ["no", "yes"], y, kind="noul", descs=["civil, or at most mildly rude", "rude, disrespectful, or likely to make someone leave the discussion"])]
            for col, txt in [("insult", "Does this comment insult someone?"), ("threat", "Does this comment contain a threat?"), ("identity_attack", "Does this comment attack someone's identity?")]:
                qs.append(Q(txt, ["no", "yes"], int(r[col] >= 0.5), kind="noul"))
            out.append(Example(_clip(r["text"], 2000), qs, t))
            if pos + neg >= cap:
                break
        return out
    return _both(conv, tr, ev, "civil_comments")


@task("toxic_chat", group="moderation")
def _tc():
    tr, ev = _pair("lmsys/toxic-chat", "toxicchat0124", "train", "test")
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            qs = [Q("Is this user message toxic or harmful?", ["no", "yes"], int(r["toxicity"]), kind="noul"),
                  Q("Is this user message a jailbreak attempt?", ["no", "yes"], int(r["jailbreaking"]), kind="noul")]
            out.append(Example(_clip(r["user_input"], 2500), qs, t))
        return out
    return _both(conv, tr, ev, "toxic_chat")


@task("aegis_safety", group="moderation")
def _aegis():
    ds = _ld("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "default", "train").shuffle(seed=SEED)
    def conv(d, t, cap):
        out = []
        for r in _sub(d, cap * 2):
            txt = r.get("prompt") or ""
            if not txt or not r.get("prompt_label"):
                continue
            y = int(str(r["prompt_label"]).lower() == "unsafe")
            out.append(Example(_clip(txt, 2500), [Q("Is this user prompt unsafe under a content-safety policy?", ["no", "yes"], y, kind="noul")], t))
            if len(out) >= cap:
                break
        return out
    n_ev = 1500
    return conv(ds.select(range(n_ev, len(ds))), "aegis_safety", TRAIN_CAP), conv(ds.select(range(n_ev)), "aegis_safety", EVAL_CAP)


@task("sms_spam", group="moderation")
def _sms():
    ds = _ld("ucirvine/sms_spam", None, "train").shuffle(seed=SEED)
    f = lambda d, t, cap: _yesno("Is this SMS message spam?", lambda r: r["sms"], lambda r: r["label"], d, t, cap)
    return f(ds.select(range(800, len(ds))), "sms_spam", TRAIN_CAP), f(ds.select(range(800)), "sms_spam", EVAL_CAP)


@task("enron_spam", group="moderation")
def _enron():
    tr, ev = _pair("SetFit/enron_spam", None, "train", "test")
    f = lambda ds, t, cap: _yesno("Is this email spam?", lambda r: f"Subject: {r['subject']}\n\n{_clip(r['text'], 2000)}", lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "enron_spam")


@task("insincere_questions", group="moderation")
def _insq():
    tr, ev = _pair("SetFit/insincere-questions", None, "train", "test")
    f = lambda ds, t, cap: _yesno("Is this question insincere (rhetorical, provocative, or not seeking a real answer)?", lambda r: r["text"], lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "insincere_questions")


@task("ade", heldout=True, group="moderation")
def _ade():
    tr, ev = _pair("SetFit/ade_corpus_v2_classification", None, "train", "test")
    f = lambda ds, t, cap: _yesno("Does this sentence report an adverse drug effect?", lambda r: r["text"], lambda r: r["label"], ds, t, cap)
    return _both(f, tr, ev, "ade")


# =============================================================== NLI / paraphrase / fact / relevance
NLI = ["entailment (text 2 follows from text 1)", "neutral (cannot tell)", "contradiction (text 2 conflicts with text 1)"]


def _nli(ds, a, b, t, cap, lab="label"):
    return _cls(ds, lambda r: f"Text 1: {r[a]}\nText 2: {r[b]}", lambda r: int(r[lab]), "What is the logical relation between text 1 and text 2?", NLI, t, cap)


@task("snli", group="nli")
def _snli():
    tr, ev = _pair("stanfordnlp/snli", None, "train", "test")
    f = lambda ds, t, cap: _nli(ds.filter(lambda r: r["label"] >= 0), "premise", "hypothesis", t, cap)
    return _both(f, tr, ev, "snli")


@task("mnli", group="nli")
def _mnli():
    tr, ev = _pair("nyu-mll/multi_nli", None, "train", "validation_matched")
    f = lambda ds, t, cap: _nli(ds, "premise", "hypothesis", t, cap)
    return _both(f, tr, ev, "mnli")


@task("anli", group="nli")
def _anli():
    tr = _ld("facebook/anli", None, "train_r3"); ev = _ld("facebook/anli", None, "test_r3")
    f = lambda ds, t, cap: _nli(ds, "premise", "hypothesis", t, cap)
    return _both(f, tr, ev, "anli")


@task("cb", heldout=True, group="nli")
def _cb():
    tr, ev = _pair("aps/super_glue", "cb", "train", "validation")
    cb2nli = {0: 0, 1: 2, 2: 1}
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Text 1: {r['premise']}\nText 2: {r['hypothesis']}", lambda r: cb2nli.get(int(r["label"]), -1), "What is the logical relation between text 1 and text 2?", NLI, t, cap)
    return _both(f, tr, ev, "cb")


@task("rte", group="nli")
def _rte():
    tr, ev = _pair("nyu-mll/glue", "rte", "train", "validation")
    f = lambda ds, t, cap: _yesno("Does text 2 follow from text 1?", lambda r: f"Text 1: {r['sentence1']}\nText 2: {r['sentence2']}", lambda r: 1 - int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "rte")


@task("qnli", group="nli")
def _qnli():
    tr, ev = _pair("nyu-mll/glue", "qnli", "train", "validation")
    f = lambda ds, t, cap: _yesno("Does the sentence contain the answer to the question?", lambda r: f"Question: {r['question']}\nSentence: {r['sentence']}", lambda r: 1 - int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "qnli")


@task("qqp", group="nli")
def _qqp():
    tr, ev = _pair("nyu-mll/glue", "qqp", "train", "validation")
    f = lambda ds, t, cap: _yesno("Are these two questions asking the same thing?", lambda r: f"Question 1: {r['question1']}\nQuestion 2: {r['question2']}", lambda r: int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "qqp")


@task("mrpc", group="nli")
def _mrpc():
    tr, ev = _pair("nyu-mll/glue", "mrpc", "train", "validation")
    f = lambda ds, t, cap: _yesno("Are these two sentences paraphrases of each other?", lambda r: f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", lambda r: int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "mrpc")


@task("paws", heldout=True, group="nli")
def _paws():
    tr, ev = _pair("google-research-datasets/paws", "labeled_final", "train", "test")
    f = lambda ds, t, cap: _yesno("Do these two sentences have the same meaning?", lambda r: f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", lambda r: int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "paws")


@task("cola", group="nli")
def _cola():
    tr, ev = _pair("nyu-mll/glue", "cola", "train", "validation")
    f = lambda ds, t, cap: _yesno("Is this sentence grammatically acceptable?", lambda r: r["sentence"], lambda r: int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "cola")


@task("wic", group="nli")
def _wic():
    tr, ev = _pair("aps/super_glue", "wic", "train", "validation")
    f = lambda ds, t, cap: _yesno("Is the word used with the same meaning in both sentences?", lambda r: f"Word: {r['word']}\nSentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", lambda r: int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "wic")


@task("fever", group="fact")
def _fever():
    tr, ev = _pair("copenlu/fever_gold_evidence", None, "train", "validation")
    names = ["supported by the evidence", "refuted by the evidence", "not enough information"]
    lab = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    def ctx(r):
        ev_txt = "\n".join(f"[{e[0]}] {e[2]}" for e in r["evidence"][:4] if len(e) >= 3)
        return f"Claim: {r['claim']}\nEvidence:\n{_clip(ev_txt, 2500)}"
    f = lambda ds, t, cap: _cls(ds, ctx, lambda r: lab.get(r["label"]), "Is the claim supported or refuted by the evidence?", names, t, cap)
    return _both(f, tr, ev, "fever")


@task("wiki_qa", group="relevance")
def _wqa():
    tr, ev = _pair("microsoft/wiki_qa", None, "train", "test")
    def conv(ds, t, cap):
        from datasets import concatenate_datasets
        pos = ds.filter(lambda r: r["label"] == 1); neg = ds.filter(lambda r: r["label"] == 0).shuffle(seed=SEED)
        n = min(len(pos), cap // 3)
        d = concatenate_datasets([pos.select(range(n)), neg.select(range(min(len(neg), 2 * n)))]).shuffle(seed=SEED)
        return _yesno("Does the candidate sentence answer the question?", lambda r: f"Question: {r['question']}\nCandidate sentence (from '{r['document_title']}'): {r['answer']}", lambda r: int(r["label"]), d, t, cap)
    return _both(conv, tr, ev, "wiki_qa")


@task("msmarco_rel", group="relevance")
def _msm():
    tr, ev = _pair("microsoft/ms_marco", "v1.1", "train", "validation")
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        for r in _sub(ds, cap):
            ps = r["passages"]; sel = [i for i, s in enumerate(ps["is_selected"]) if s == 1]
            if not sel:
                continue
            i = sel[0] if rng.random() < 0.5 else rng.choice([j for j in range(len(ps["passage_text"])) if j not in sel] or sel)
            out.append(Example(f"Query: {r['query']}\nPassage: {_clip(ps['passage_text'][i], 1500)}", [Q("Does the passage answer the query?", ["no", "yes"], int(i in sel), kind="noul")], t))
        return out
    return _both(conv, tr, ev, "msmarco_rel")


@task("msmarco_rank", group="relevance")
def _msm_rank():
    """Which of the candidate passages best answers the query? (choice over passages)"""
    tr, ev = _pair("microsoft/ms_marco", "v1.1", "train", "validation")
    def conv(ds, t, cap):
        rng = random.Random(SEED + 1); out = []
        for r in _sub(ds, cap * 2, seed=SEED + 1):
            ps = r["passages"]; sel = [i for i, s in enumerate(ps["is_selected"]) if s == 1]
            if len(sel) != 1 or len(ps["passage_text"]) < 3:
                continue
            idx = list(range(len(ps["passage_text"])))[:6]
            if sel[0] not in idx:
                continue
            opts = [_clip(ps["passage_text"][i], 400) for i in idx]
            out.append(Example(f"Query: {r['query']}", [Q("Which passage best answers the query?", opts, idx.index(sel[0]))], t))
            if len(out) >= cap:
                break
        return out
    return _both(conv, tr, ev, "msmarco_rank")


@task("multirc", group="relevance")
def _multirc():
    tr, ev = _pair("aps/super_glue", "multirc", "train", "validation")
    f = lambda ds, t, cap: _yesno("Is the candidate answer correct according to the paragraph?", lambda r: f"Paragraph: {_clip(r['paragraph'], 2500)}\nQuestion: {r['question']}\nCandidate answer: {r['answer']}", lambda r: int(r["label"]), ds, t, cap)
    return _both(f, tr, ev, "multirc")


# =============================================================== QA / reading comprehension
def _mcq(ds, ctx_fn, opts_fn, gold_fn, q, t, cap):
    out = []
    for r in _sub(ds, cap):
        try:
            opts = [str(o).strip() for o in opts_fn(r)]
            g = gold_fn(r)
        except Exception:
            continue
        if g is None or g < 0 or g >= len(opts) or len(opts) < 2:
            continue
        ctx = ctx_fn(r)
        if not str(ctx).strip():
            continue
        out.append(Example(str(ctx).strip(), [Q(q, opts, int(g))], t))
    return out


def _key_idx(r, key="answerKey"):
    labels = r["choices"]["label"]; k = r[key]
    return labels.index(k) if k in labels else None


@task("boolq", group="qa")
def _boolq():
    tr, ev = _pair("google/boolq", None, "train", "validation")
    f = lambda ds, t, cap: _yesno("Based on the passage, what is the answer to the question?", lambda r: f"Passage: {_clip(r['passage'], 2500)}\nQuestion: {r['question']}", lambda r: int(r["answer"]), ds, t, cap)
    return _both(f, tr, ev, "boolq")


@task("strategyqa", heldout=True, group="qa")
def _sqa():
    tr, ev = _pair("ChilleD/StrategyQA", None, "train", "test")
    f = lambda ds, t, cap: _yesno("What is the answer to this question?", lambda r: r["question"], lambda r: int(r["answer"]), ds, t, cap)
    return _both(f, tr, ev, "strategyqa")


@task("pubmedqa", heldout=True, group="qa")
def _pmq():
    ds = _ld("qiaojin/PubMedQA", "pqa_labeled", "train").shuffle(seed=SEED)
    names = ["yes", "no", "maybe"]
    f = lambda d, t, cap: _cls(d, lambda r: ("Abstract: " + " ".join(r["context"]["contexts"]))[:2500] + f"\nQuestion: {r['question']}", lambda r: names.index(r["final_decision"]), "Based on the abstract, what is the answer?", names, t, cap)
    return [], f(ds.select(range(600)), "pubmedqa", EVAL_CAP)


@task("arc", group="qa")
def _arc():
    from datasets import concatenate_datasets
    tr = concatenate_datasets([_ld("allenai/ai2_arc", "ARC-Challenge", "train"), _ld("allenai/ai2_arc", "ARC-Easy", "train")])
    ev = _ld("allenai/ai2_arc", "ARC-Challenge", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "arc")


@task("commonsense_qa", group="qa")
def _csqa():
    tr, ev = _pair("tau/commonsense_qa", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "commonsense_qa")


@task("qasc", group="qa")
def _qasc():
    tr, ev = _pair("allenai/qasc", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "qasc")


@task("openbookqa", group="qa")
def _obqa():
    tr, ev = _pair("allenai/openbookqa", "main", "train", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question_stem"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly completes or answers the question?", t, cap)
    return _both(f, tr, ev, "openbookqa")


@task("sciq", heldout=True, group="qa")
def _sciq():
    tr, ev = _pair("allenai/sciq", None, "train", "validation")
    def opts(r):
        o = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        rng = random.Random(zlib.crc32(r["question"].encode()) & 0xffff); perm = list(range(4)); rng.shuffle(perm); r["_perm"] = perm  # PORT: crc32, was hash() (PYTHONHASHSEED-dependent)
        return [o[i] for i in perm]
    f = lambda ds, t, cap: _mcq(ds, lambda r: (f"Support: {r['support'][:1500]}\n" if r["support"] else "") + f"Question: {r['question']}", opts, lambda r: r["_perm"].index(0), "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "sciq")


@task("hellaswag", group="qa")
def _hs():
    tr, ev = _pair("Rowan/hellaswag", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["ctx"], lambda r: r["endings"], lambda r: int(r["label"]) if r["label"] != "" else None, "Which ending is the most plausible continuation?", t, cap)
    return _both(f, tr, ev, "hellaswag")


@task("piqa", group="qa")
def _piqa():
    tr, ev = _pair("ybisk/piqa", None, "train", "validation", rev="refs/convert/parquet")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["goal"], lambda r: [r["sol1"], r["sol2"]], lambda r: r["label"], "Which solution is the correct way to achieve the goal?", t, cap)
    return _both(f, tr, ev, "piqa")


@task("social_iqa", heldout=True, group="qa")
def _siqa():
    tr, ev = _pair("allenai/social_i_qa", None, "train", "validation", rev="refs/convert/parquet")
    f = lambda ds, t, cap: _mcq(ds, lambda r: f"{r['context']}\nQuestion: {r['question']}", lambda r: [r["answerA"], r["answerB"], r["answerC"]], lambda r: int(r["label"]) - 1, "Which answer is most appropriate?", t, cap)
    return _both(f, tr, ev, "social_iqa")


@task("winogrande", group="qa")
def _wg():
    tr, ev = _pair("allenai/winogrande", "winogrande_xl", "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["sentence"], lambda r: [r["option1"], r["option2"]], lambda r: int(r["answer"]) - 1 if r["answer"] in ("1", "2") else None, "Which option correctly fills the blank (_)?", t, cap)
    return _both(f, tr, ev, "winogrande")


@task("copa", group="qa")
def _copa():
    tr, ev = _pair("aps/super_glue", "copa", "train", "validation")
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            q = "What was the cause of this?" if r["question"] == "cause" else "What happened as a result?"
            out.append(Example(r["premise"], [Q(q, [r["choice1"], r["choice2"]], int(r["label"]))], t))
        return out
    return _both(conv, tr, ev, "copa")


@task("race", group="qa")
def _race():
    tr, ev = _pair("ehovy/race", "all", "train", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: f"Article: {_clip(r['article'], 3000)}\nQuestion: {r['question']}", lambda r: r["options"], lambda r: "ABCD".index(r["answer"]), "Which option correctly answers the question about the article?", t, cap)
    return _both(f, tr, ev, "race")


@task("mmlu", group="qa")
def _mmlu():
    tr = _ld("cais/mmlu", "all", "auxiliary_train"); ev = _ld("cais/mmlu", "all", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: _clip(r["question"], 2000), lambda r: r["choices"], lambda r: int(r["answer"]), "Which option is correct?", t, cap)
    return _both(f, tr, ev, "mmlu")


@task("mmlu_pro", group="qa")
def _mmlu_pro():
    tr, ev = _pair("TIGER-Lab/MMLU-Pro", None, "validation", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: _clip(r["question"], 2000), lambda r: r["options"], lambda r: int(r["answer_index"]), "Which option is correct?", t, cap)
    return f(ev.shuffle(seed=SEED).select(range(min(len(ev), 4000), len(ev))), "mmlu_pro", TRAIN_CAP), f(ev.shuffle(seed=SEED).select(range(min(len(ev), 4000))), "mmlu_pro", EVAL_CAP)


@task("medmcqa", group="qa")
def _medmcqa():
    tr, ev = _pair("openlifescienceai/medmcqa", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: [r["opa"], r["opb"], r["opc"], r["opd"]], lambda r: int(r["cop"]), "Which option is correct?", t, cap)
    return _both(f, tr, ev, "medmcqa")


@task("truthfulqa", heldout=True, group="qa")
def _tqa():
    ds = _ld("truthfulqa/truthful_qa", "multiple_choice", "validation")
    f = lambda d, t, cap: _mcq(d, lambda r: r["question"], lambda r: r["mc1_targets"]["choices"], lambda r: r["mc1_targets"]["labels"].index(1), "Which answer is true?", t, cap)
    return [], f(ds, "truthfulqa", EVAL_CAP)


@task("xstory_cloze", heldout=True, group="qa")
def _xsc():
    ds = _ld("juletxara/xstory_cloze", "en", "eval")
    return [], _mcq(ds, lambda r: " ".join(r[f"input_sentence_{i}"] for i in range(1, 5)), lambda r: [r["sentence_quiz1"], r["sentence_quiz2"]], lambda r: int(r["answer_right_ending"]) - 1, "Which sentence is the right ending of the story?", "xstory_cloze", EVAL_CAP)


@task("quality", heldout=True, group="qa")
def _quality():
    tr, ev = _pair("emozilla/quality", None, "train", "validation")
    def conv(ds, t, cap):
        base = min(int(r["answer"]) for r in ds)
        return _mcq(ds, lambda r: f"Article: {_clip(r['article'], 6000)}\nQuestion: {r['question']}", lambda r: list(r["options"]), lambda r: int(r["answer"]) - base, "Which option correctly answers the question about the article?", t, cap)
    return [], conv(ev, "quality", EVAL_CAP)


# =============================================================== pairwise preference / judging
def _pair_ex(ctx, r1, r2, better, t, rng, tie=False):
    flip = rng.random() < 0.5
    a, b = (r2, r1) if flip else (r1, r2)
    g = better if better == 2 or not flip else 1 - better
    opts = ["response A", "response B"] + (["about equally good"] if tie else [])
    return Example(f"{ctx}\n\nResponse A:\n{a}\n\nResponse B:\n{b}", [Q("Which response is the better answer to the prompt?", opts, g)], t)


@task("ultrafeedback_pref", group="judge")
def _uf():
    tr, ev = _pair("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "test_prefs")
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        ds = ds.filter(lambda r: r["score_chosen"] > r["score_rejected"])
        for r in _sub(ds, cap):
            c = r["chosen"][-1]["content"]; j = r["rejected"][-1]["content"]
            out.append(_pair_ex(f"Prompt:\n{_clip(r['prompt'], 2000)}", _clip(c, 1800), _clip(j, 1800), 0, t, rng))
        return out
    return _both(conv, tr, ev, "ultrafeedback_pref")


@task("helpsteer3_pref", group="judge")
def _hs3():
    tr, ev = _pair("nvidia/HelpSteer3", "preference", "train", "validation")
    legend = {-3: "response 1 is much better", -2: "response 1 is better", -1: "response 1 is slightly better", 0: "both are about the same", 1: "response 2 is slightly better", 2: "response 2 is better", 3: "response 2 is much better"}
    def conv(ds, t, cap):
        ds = ds.filter(lambda r: str(r["language"]).lower().startswith("en"))
        out = []
        keys, proto = _scale_q("Which response is the better final reply, and by how much?", legend)
        for r in _sub(ds, cap):
            conv_txt = "\n".join(f"{m['role']}: {_clip(m['content'], 1200)}" for m in r["context"][-4:])
            ctx = f"Conversation:\n{_clip(conv_txt, 3000)}\n\nResponse 1:\n{_clip(r['response1'], 1800)}\n\nResponse 2:\n{_clip(r['response2'], 1800)}"
            out.append(Example(ctx, [Q(proto.text, proto.options, keys.index(int(r["overall_preference"])), kind="score", descs=proto.descs)], t))
        return out
    return _both(conv, tr, ev, "helpsteer3_pref")


@task("hh_rlhf", group="judge")
def _hh():
    tr, ev = _pair("Anthropic/hh-rlhf", None, "train", "test")
    def split_last(s):
        i = s.rfind("\n\nAssistant:")
        return s[:i].strip(), s[i + len("\n\nAssistant:"):].strip()
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        for r in _sub(ds, cap):
            p1, c = split_last(r["chosen"]); p2, j = split_last(r["rejected"])
            if p1 != p2 or not c or not j or c == j:
                continue
            out.append(_pair_ex(f"Conversation:\n{_clip(p1, 2500)}", _clip(c, 1500), _clip(j, 1500), 0, t, rng))
        return out
    return conv(tr, "hh_rlhf", 8000), conv(ev, "hh_rlhf", EVAL_CAP)


@task("arena_pref", heldout=True, group="judge")
def _arena():
    ds = _ld("lmarena-ai/arena-human-preference-55k", None, "train")
    def conv(d, t, cap):
        rng = random.Random(SEED); out = []
        for r in _sub(d, cap * 3):
            try:
                ps = json.loads(r["prompt"]); ra = json.loads(r["response_a"]); rb = json.loads(r["response_b"])
            except Exception:
                continue
            if len(ps) != 1 or not ra[0] or not rb[0]:
                continue
            better = 0 if r["winner_model_a"] else (1 if r["winner_model_b"] else 2)
            out.append(_pair_ex(f"Prompt:\n{_clip(ps[0], 2000)}", _clip(ra[0], 1800), _clip(rb[0], 1800), better, t, rng, tie=True))
            if len(out) >= cap:
                break
        return out
    return [], conv(ds, "arena_pref", EVAL_CAP)


@task("reward_bench", heldout=True, group="judge")
def _rb():
    ds = _ld("allenai/reward-bench", "default", "filtered")
    rng = random.Random(SEED)
    return [], [_pair_ex(f"Prompt:\n{_clip(r['prompt'], 2000)}", _clip(r["chosen"], 1800), _clip(r["rejected"], 1800), 0, "reward_bench", rng) for r in _sub(ds, EVAL_CAP)]


@task("mt_bench_judge", group="judge")
def _mtb():
    ds = _ld("lmsys/mt_bench_human_judgments", None, "human")
    rng = random.Random(SEED); out = []
    for r in ds:
        if r["turn"] != 1:
            continue
        w = r["winner"]
        better = 0 if w == "model_a" else (1 if w == "model_b" else 2)
        ca = r["conversation_a"]; cb = r["conversation_b"]
        if len(ca) < 2 or len(cb) < 2:
            continue
        out.append(_pair_ex(f"Prompt:\n{_clip(ca[0]['content'], 2000)}", _clip(ca[1]["content"], 1800), _clip(cb[1]["content"], 1800), better, "mt_bench_judge", rng, tie=True))
    rng.shuffle(out); n_ev = min(400, len(out) // 5)
    return out[n_ev:], out[:n_ev]


# =============================================================== tool / function selection
pool_desc = {}


def _tool_ex(user_msg, funcs, gold_name, pool, rng, t, k=6):
    names = list(funcs)
    while len(names) < k - 1 and pool:
        n = rng.choice(pool)
        if n not in names:
            names.append(n)
    rng.shuffle(names)
    tools = [{"name": n, "description": _clip(funcs.get(n) or pool_desc.get(n, ""), 200)} for n in names]
    ctx = json.dumps({"available_tools": tools, "user_request": _clip(user_msg, 1500)}, ensure_ascii=False, indent=1)
    opts = names + [NONE_OPT + " (no function call needed)"]
    g = opts.index(gold_name) if gold_name in opts else len(opts) - 1
    return Example(ctx, [Q("Which function should be called to handle this request?", opts, g),
                         Q("Does this request require calling a function at all?", ["no", "yes"], int(gold_name is not None and gold_name in funcs), kind="noul")], t)


@task("glaive_tools", group="tools")
def _glaive():
    ds = _ld("glaiveai/glaive-function-calling-v2", None, "train").shuffle(seed=SEED)
    rx_obj = re.compile(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"', re.S)
    def parse(r):
        funcs = dict(rx_obj.findall(r["system"])); chat = r["chat"]
        m = re.search(r"USER:\s*(.*?)\n\n\nASSISTANT:\s*(.*?)(?:<\|endoftext\|>|\n\n\n)", chat, re.S)
        if not m or not funcs:
            return None
        user, asst = m.group(1).strip(), m.group(2).strip()
        if asst.startswith("<functioncall>"):
            mm = re.search(r'"name":\s*"([^"]+)"', asst); gold = mm.group(1) if mm else None
            if gold not in funcs:
                return None
        else:
            gold = None
        return user, funcs, gold
    parsed = [p for p in (parse(r) for r in ds.select(range(min(60000, len(ds))))) if p]
    for _, f, _ in parsed:
        pool_desc.update(f)
    pool = list(pool_desc)
    def conv(items, t, cap):
        rng = random.Random(SEED)
        calls = [p for p in items if p[2]]; nones = [p for p in items if not p[2]]
        sel = calls[: int(cap * 0.75)] + nones[: cap - int(cap * 0.75)]; rng.shuffle(sel)
        return [_tool_ex(u, f, g, pool, rng, t) for u, f, g in sel]
    return conv(parsed[2000:], "glaive_tools", TRAIN_CAP), conv(parsed[:2000], "glaive_tools", EVAL_CAP)


@task("toolace", group="tools")
def _toolace():
    ds = _ld("Team-ACE/ToolACE", None, "train").shuffle(seed=SEED)
    def parse(r):
        i = r["system"].find("you can invoke:")
        if i < 0:
            return None
        try:
            fl, _ = json.JSONDecoder().raw_decode(r["system"][i + len("you can invoke:"):].lstrip())
        except Exception:
            return None
        funcs = {f["name"]: f.get("description", "") for f in fl if "name" in f}
        conv = r["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            return None
        a = conv[1]["value"].strip()
        if a.startswith("["):
            mm = re.match(r"\[\s*([^(\]]+?)\s*\(", a); gold = mm.group(1).strip() if mm else None
            if gold not in funcs:
                return None
        else:
            gold = None
        return conv[0]["value"], funcs, gold
    parsed = [p for p in (parse(r) for r in ds) if p]
    for _, f, _ in parsed:
        pool_desc.update(f)
    pool = list(pool_desc)
    def conv(items, t, cap):
        rng = random.Random(SEED)
        return [_tool_ex(u, f, g, pool, rng, t, k=8) for u, f, g in items[:cap]]
    return conv(parsed[1000:], "toolace", TRAIN_CAP), conv(parsed[:1000], "toolace", EVAL_CAP)


@task("hermes_tools", heldout=True, group="tools")
def _hermes():
    ds = _ld("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train")
    def parse(r):
        try:
            tl = json.loads(r["tools"])
        except Exception:
            return None
        funcs = {t["function"]["name"]: t["function"].get("description", "") for t in tl if "function" in t}
        hum = next((c["value"] for c in r["conversations"] if c["from"] == "human"), None)
        gpt = next((c["value"] for c in r["conversations"] if c["from"] == "gpt"), "")
        mm = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gpt, re.S); gold = None
        if mm:
            try:
                gold = json.loads(mm.group(1)).get("name")
            except Exception:
                m2 = re.search(r"'name':\s*'([^']+)'", mm.group(1)); gold = m2.group(1) if m2 else None
        if not hum or not funcs or (gold and gold not in funcs):
            return None
        return hum, funcs, gold
    parsed = [p for p in (parse(r) for r in ds) if p]
    pool = list({n for _, f, _ in parsed for n in f}); pool_desc.update({n: d for _, f, _ in parsed for n, d in f.items()})
    rng = random.Random(SEED)
    return [], [_tool_ex(u, f, g, pool, rng, "hermes_tools", k=8) for u, f, g in parsed[:EVAL_CAP]]


# =============================================================== situation -> action (agents)
@task("agenttraj", group="agent")
def _agenttraj():
    ds = _ld("AgentGym/AgentTraj-L", None, "train")
    rng = random.Random(SEED); out = []
    rx = re.compile(r"Action:\s*(.+?)\s*$", re.S)
    pool = defaultdict(list); trajs = []
    for r in ds:
        env = r["item_id"].split("_")[0]; conv = r["conversations"]; steps = []
        for i, m in enumerate(conv):
            if m["from"] == "gpt":
                mm = rx.search(m["value"])
                if mm:
                    act = mm.group(1).strip().split("\n")[0][:120]; steps.append((i, act)); pool[env].append(act)
        if len(steps) >= 2:
            trajs.append((env, conv, steps))
    rng.shuffle(trajs)
    for env, conv, steps in trajs:
        task_txt = _clip(conv[2]["value"] if len(conv) > 2 and conv[2]["from"] == "human" else conv[0]["value"], 1500)
        own = [a for _, a in steps]
        for k, (i, act) in enumerate(steps):
            if rng.random() > 0.35:
                continue
            hist = [m for m in conv[:i] if m["from"] == "human"][-2:]
            obs = "\n".join(_clip(m["value"], 700) for m in hist[-1:])
            prev = [a for _, a in steps[:k]][-3:]
            ctx = json.dumps({"environment": env, "task": task_txt, "recent_actions": prev, "latest_observation": obs}, ensure_ascii=False, indent=1)
            distract = list(dict.fromkeys(a for a in own if a != act)); rng.shuffle(distract); cands = distract[:3]
            others = list(dict.fromkeys(a for a in pool[env] if a != act and a not in cands))
            while len(cands) < 4 and others:
                cands.append(others.pop(rng.randrange(len(others))))
            opts = [act] + cands; rng.shuffle(opts)
            out.append(Example(ctx, [Q("Which action should the agent take next?", opts, opts.index(act))], "agenttraj"))
        if len(out) >= TRAIN_CAP + EVAL_CAP:
            break
    return out[EVAL_CAP:], out[:EVAL_CAP]


def _elem_desc(c):
    try:
        at = json.loads(c["attributes"])
    except Exception:
        at = {}
    bits = [c.get("tag", "")]
    for k in ("aria_label", "aria-label", "title", "alt", "placeholder", "name", "value", "type", "role", "id", "class"):
        if at.get(k):
            bits.append(f"{k}={str(at[k])[:40]}")
    return " ".join(bits)[:120]


@task("mind2web", group="agent")
def _m2w():
    ds = _ld("osunlp/Mind2Web", None, "train")
    rng = random.Random(SEED); out = []
    ops = ["CLICK", "TYPE", "SELECT"]
    for r in ds:
        prev = []
        for a, rep in zip(r["actions"], r["action_reprs"]):
            if a["pos_candidates"]:
                pos = _elem_desc(a["pos_candidates"][0]); negs = [_elem_desc(c) for c in a["neg_candidates"]]
                negs = [n for n in negs if n != pos]; rng.shuffle(negs); opts = [pos] + negs[:5]; rng.shuffle(opts)
                op = a["operation"]["op"]; val = a["operation"].get("value") or ""
                ctx = json.dumps({"website": r["website"], "domain": r["domain"], "task": r["confirmed_task"], "actions_so_far": prev[-4:]}, ensure_ascii=False, indent=1)
                qs = [Q("Which page element should the next action target?", opts, opts.index(pos))]
                if op in ops:
                    qs.append(Q("What kind of operation is the next action?", ops, ops.index(op)))
                out.append(Example(ctx, qs, "mind2web"))
            prev.append(rep)
        if len(out) >= TRAIN_CAP + EVAL_CAP:
            break
    rng.shuffle(out)
    return out[EVAL_CAP:], out[:EVAL_CAP]


# =============================================================== multi-attribute
@task("bias_in_bios", group="misc")
def _bib():
    tr, ev = _pair("LabHC/bias_in_bios", None, "train", "test")
    profs = ["accountant", "architect", "attorney", "chiropractor", "comedian", "composer", "dentist", "dietitian", "dj", "filmmaker", "interior designer", "journalist", "model", "nurse", "painter", "paralegal", "pastor", "personal trainer", "photographer", "physician", "poet", "professor", "psychologist", "rapper", "software engineer", "surgeon", "teacher", "yoga teacher"]
    def conv(ds, t, cap):
        return [Example(_clip(r["hard_text"], 2000), [Q("What is this person's profession?", profs, int(r["profession"]))], t) for r in _sub(ds, cap)]
    return _both(conv, tr, ev, "bias_in_bios")


# ════════════════════════════════════════════════════════════════════════════════════════════════
# END adapted from system-one-open. Everything below is ours.
# ════════════════════════════════════════════════════════════════════════════════════════════════

SOURCES: list = []          # every _ld() call a loader makes, recorded for the manifest
_ld_upstream = _ld


def _ld(name, cfg=None, split=None, rev=None):  # noqa: F811 — wraps the ported loader helper
    key = (name, cfg, rev)
    if key not in SOURCES:
        SOURCES.append(key)
    return _ld_upstream(name, cfg, split, rev)


# ── jev tasks ────────────────────────────────────────────────────────────────────────────────
JEV_V0_FAMILY = {"injection": "safety", "sentiment": "sentiment", "routing": "topic", "langid": "language"}
JEV_SOURCE = {
    "injection": ("deepset/prompt-injections", None), "sentiment": ("stanfordnlp/sst2", None),
    "routing": ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None),
    "langid": ("papluca/language-identification", None),
    "offensive": ("cardiffnlp/tweet_eval", "offensive"), "hate": ("cardiffnlp/tweet_eval", "hate"),
    "spam": ("ucirvine/sms_spam", None), "tweetsent": ("cardiffnlp/tweet_eval", "sentiment"),
    "emotion": ("dair-ai/emotion", None), "agnews": ("fancyzhx/ag_news", None),
    "dbpedia": ("fancyzhx/dbpedia_14", None), "yahoo": ("community-datasets/yahoo_answers_topics", None),
    "finnews": ("zeroshot/twitter-financial-news-sentiment", None),
    "massive": ("mteb/amazon_massive_intent", "en"),
    "rte": ("nyu-mll/glue", "rte"), "mrpc": ("nyu-mll/glue", "mrpc"), "qnli": ("nyu-mll/glue", "qnli"),
    "cola": ("nyu-mll/glue", "cola"), "snli": ("stanfordnlp/snli", None),
    "yelpstars": ("Yelp/yelp_review_full", None), "sst5": ("SetFit/sst5", None),
    "appreviews": ("sealuzh/app_reviews", None),
}
JEV_SCORE_LEVELS = {
    "yelpstars": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    "appreviews": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    "sst5": ["1 - very negative", "2 - negative", "3 - neutral", "4 - positive", "5 - very positive"],
}
DEFAULT_JEV_HELDOUT = "language,ordinal"


# ── helpers ──────────────────────────────────────────────────────────────────────────────────
TRUNC_MARK = "\n[... truncated ...]\n"


def stable_rng(*parts) -> random.Random:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def clean_state(s, max_chars, stats):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    s = s.strip()
    if not s:
        stats["drop_empty_state"] += 1
        return None
    if len(s) > max_chars:
        stats["state_truncated"] += 1
        budget = max_chars - len(TRUNC_MARK)
        head = int(budget * 0.7)
        s = s[:head] + TRUNC_MARK + s[len(s) - (budget - head):]
    return s


def _is_none_opt(o: str) -> bool:
    return o.lower().startswith("none of the above")


def convert_q(q: Q, rowkey: str, stats: Counter):
    """system-one-open Q -> (DATA-FORMAT question dict, gold outcome) or None."""
    opts = [str(o).strip() for o in q.options]
    descs = list(q.descs) if q.descs else [None] * len(opts)
    descs = [(str(d).strip() or None) if d is not None else None for d in descs]
    g = q.gold
    if g is None or not 0 <= g < len(opts):
        stats["drop_no_gold"] += 1
        return None
    instr = str(q.text).strip()
    if q.kind == "noul":
        # upstream noul is always options ["no", "yes"]; gold 1 == yes
        assert [o.lower() for o in opts] == ["no", "yes"], opts
        qd = {"type": "noul", "instructions": instr}
        if any(descs):
            qd["criteria"] = {"true": descs[1], "false": descs[0]}
        return qd, ("yes" if g == 1 else "no")
    if q.kind == "score":
        if len(opts) > MAX_SCORE:
            stats["drop_score_too_many_levels"] += 1
            return None
        levels = []
        for o, d in zip(opts, descs):
            if not d:
                levels.append(o)
            else:
                try:
                    float(o)
                    levels.append(d)            # numeric key ("0".."5", "-3".."3"): the desc says it all
                except ValueError:
                    levels.append(f"{o}: {d}")  # named level (e.g. ticket priority "low")
        return {"type": "score", "instructions": instr, "criteria": levels}, str(g)
    # choice
    idx = list(range(len(opts)))
    if not opts[g]:
        stats["drop_empty_gold_option"] += 1
        return None
    # duplicate / empty option names can't be dict keys: drop non-gold ones, drop the q if gold is ambiguous
    seen, keep = {}, []
    for i in idx:
        o = opts[i]
        if not o:
            stats["option_empty_removed"] += 1
            continue
        if o in seen:
            if i == g or seen[o] == g:
                stats["drop_gold_duplicated"] += 1
                return None
            stats["option_duplicate_removed"] += 1
            continue
        seen[o] = i
        keep.append(i)
    idx = keep
    if len(idx) > MAX_CHOICE:
        stats["choice_reduced_to_20"] += 1
        must = [i for i in idx if i == g or _is_none_opt(opts[i])]
        rest = [i for i in idx if i not in must]
        pick = set(must) | set(stable_rng("subset", rowkey).sample(rest, MAX_CHOICE - len(must)))
        idx = [i for i in idx if i in pick]           # keep the dataset's option order
    if len(idx) < 2:
        stats["drop_choice_lt2"] += 1
        return None
    crit = {opts[i]: descs[i] for i in idx}
    return {"type": "choice", "instructions": instr, "criteria": crit}, opts[g]


def classify_error(e: BaseException) -> str:
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if isinstance(e, (ModuleNotFoundError, ImportError)):
        return "missing package: " + msg[:200]
    if "gated" in low or "401" in low or "403" in low or "access to" in low:
        return "gated / no access: " + msg[:200]
    if "dataset script" in low or "trust_remote_code" in low or "scripts are no longer supported" in low:
        return "needs a dataset script (unsupported in datasets 4.x): " + msg[:200]
    if "doesn't exist" in low or "not found" in low or "404" in low:
        return "dataset not found: " + msg[:200]
    return "load/convert error: " + msg[:300]


_CARD_CACHE: dict = {}


def hf_card(repo: str) -> dict:
    if repo in _CARD_CACHE:
        return _CARD_CACHE[repo]
    out = {"license": "unknown", "sha": None, "gated": None}
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo)
        cd = info.card_data.to_dict() if info.card_data else {}
        lic = cd.get("license")
        if isinstance(lic, list):
            lic = "+".join(map(str, lic))
        out = {"license": lic or "unknown", "sha": info.sha, "gated": info.gated}
    except Exception as e:  # card lookup is best-effort
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    _CARD_CACHE[repo] = out
    return out


def sources_meta(sources):
    out = []
    for name, cfg, rev in sources:
        c = hf_card(name)
        out.append({"id": name, "config": cfg, "revision_used": rev or "main", "sha": c.get("sha"),
                    "license": c.get("license", "unknown"), "gated": c.get("gated")})
    return out


def write_part(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ── s1o worker (one task per subprocess) ────────────────────────────────────────────────────
def run_worker(task, a):
    global TRAIN_CAP, EVAL_CAP
    info = TASKS[task]
    TRAIN_CAP = a.train_cap
    EVAL_CAP = a.heldout_test_cap if info["heldout"] else a.calib_cap + a.test_cap
    res_path = os.path.join(a.parts_dir, f"{task}.result.json")
    t0 = time.time()
    res = {"task": task, "suite": "s1o", "family": info["group"], "heldout": info["heldout"]}
    try:
        tr, ev = info["loader"]()
    except BaseException as e:  # noqa: BLE001 — anything here means "skip this task"
        res.update(ok=False, error=classify_error(e), sources=[list(s) for s in SOURCES],
                   seconds=round(time.time() - t0, 1))
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"[{task}] FAILED {res['error']}", flush=True)
        return 1
    stats = Counter()
    if info["heldout"]:
        res["upstream_train_examples_discarded"] = len(tr)
        splits = {"test": ev}
    else:
        ev = list(ev)
        stable_rng(a.seed, task, "evalsplit").shuffle(ev)
        half = len(ev) // 2
        splits = {"train": tr, "calib": ev[:half], "test": ev[half:]}
    rows = []
    for split, exs in splits.items():
        for ei, ex in enumerate(exs):
            state = clean_state(ex.state, a.max_state_chars, stats)
            if state is None:
                continue
            for qi, q in enumerate(ex.qs):
                conv = convert_q(q, f"{task}|{split}|{ei}|{qi}|{state[:200]}", stats)
                if conv is None:
                    continue
                qd, gold = conv
                rows.append({"split": split, "ex": ei, "q": qi, "state": state, "question": qd, "gold": gold})
    write_part(os.path.join(a.parts_dir, f"{task}.jsonl"), rows)
    res.update(ok=True, sources=sources_meta(SOURCES), stats=dict(stats),
               raw_rows=dict(Counter(r["split"] for r in rows)),
               raw_examples={k: len(v) for k, v in splits.items()}, seconds=round(time.time() - t0, 1))
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"[{task}] ok {res['raw_rows']} {res['seconds']}s stats={dict(stats)}", flush=True)
    return 0


# ── jev conversion ───────────────────────────────────────────────────────────────────────────
def build_jev(a, heldout_fams):
    built = []
    for mp in sorted(f for f in os.listdir(JEV_DATA) if f.endswith(".meta.json")):
        meta = json.load(open(os.path.join(JEV_DATA, mp)))
        name = meta["suite"]
        task = f"jev_{name}"
        fam = meta.get("family") or JEV_V0_FAMILY[name]
        heldout = fam in heldout_fams
        options = meta["options"]
        stats = Counter()
        if name in JEV_SCORE_LEVELS:
            levels = JEV_SCORE_LEVELS[name]
            assert options == [str(i) for i in range(1, len(levels) + 1)], (name, options)
            mk = lambda lab: ({"type": "score", "instructions": meta["question"], "criteria": levels},
                              str(options.index(lab)))
        elif sorted(options) == ["no", "yes"]:
            mk = lambda lab: ({"type": "noul", "instructions": meta["question"]}, lab)
        else:
            assert 2 <= len(options) <= MAX_CHOICE, (name, len(options))
            mk = lambda lab: ({"type": "choice", "instructions": meta["question"],
                               "criteria": {o: None for o in options}}, lab)
        rows = []
        for split in ("train", "calib", "test"):
            if heldout and split == "train":
                continue
            for ei, line in enumerate(open(os.path.join(JEV_DATA, f"{name}.{split}.jsonl"))):
                r = json.loads(line)
                if r["label"] not in options:
                    stats["drop_label_not_in_options"] += 1
                    continue
                state = clean_state(r["text"], a.max_state_chars, stats)
                if state is None:
                    continue
                qd, gold = mk(r["label"])
                rows.append({"split": split, "ex": ei, "q": 0, "state": state, "question": qd, "gold": gold,
                             "src_id": r["id"]})
        write_part(os.path.join(a.parts_dir, f"{task}.jsonl"), rows)
        src, cfg = JEV_SOURCE[name]
        res = {"task": task, "suite": "jev", "family": fam, "heldout": heldout, "ok": True,
               "sources": sources_meta([(src, cfg, None)]), "stats": dict(stats),
               "raw_rows": dict(Counter(r["split"] for r in rows)), "jev_meta_sizes": meta["sizes"],
               "note": "built by experiments/jev/prep_data*.py (seed 20260919); splits kept as-is"
                       + ("; held-out family: train split dropped, calib+test kept" if heldout else "")}
        json.dump(res, open(os.path.join(a.parts_dir, f"{task}.result.json"), "w"), indent=1)
        built.append(task)
    return built


# ── decontamination ──────────────────────────────────────────────────────────────────────────
_LABEL_RE = re.compile(r"^\s*[^:\n]{1,60}:\s+")
_NON_ALNUM = re.compile(r"[\W_]+", re.U)
FP_LONG = 200


def fingerprint(state: str):
    """Format-insensitive key: drop line labels ("Text 1:", "Question:"), lowercase, alnum only.
    JSON-shaped states (tools / agent / helpsteer2) are skipped: their boilerplate collides."""
    if state.lstrip()[:1] in "{[":
        return None
    parts = []
    for line in state.split("\n"):
        for _ in range(2):
            line = _LABEL_RE.sub("", line, count=1)
        parts.append(_NON_ALNUM.sub("", line.lower()))
    fp = "".join(parts)
    return fp if len(fp) >= 12 else None


def read_part(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


# ── finalize: decontam, cap, write ──────────────────────────────────────────────────────────
def finalize(a, results, heldout_fams):
    os.makedirs(a.out, exist_ok=True)
    ok = [t for t, r in results.items() if r.get("ok")]
    longs, exacts = {}, {}
    for t in ok:
        for r in read_part(os.path.join(a.parts_dir, f"{t}.jsonl")):
            if r["split"] == "train":
                continue
            fp = fingerprint(r["state"])
            if fp is None:
                continue
            if len(fp) >= FP_LONG:
                longs.setdefault(fp[:FP_LONG], t)
            else:
                exacts.setdefault(fp, t)
    print(f"decontam index: {len(longs)} long + {len(exacts)} short eval fingerprints", flush=True)

    caps = {"train": a.train_cap, "calib": a.calib_cap, "test": a.test_cap}
    manifest_tasks = {}
    for t in sorted(ok):
        res = results[t]
        rows = list(read_part(os.path.join(a.parts_dir, f"{t}.jsonl")))
        contam = Counter()
        kept = []
        for r in rows:
            if r["split"] == "train":
                fp = fingerprint(r["state"])
                hit = None
                if fp is not None:
                    hit = longs.get(fp[:FP_LONG]) if len(fp) >= FP_LONG else exacts.get(fp)
                if hit is not None:
                    contam[hit] += 1
                    continue
            kept.append(r)
        by = defaultdict(lambda: defaultdict(list))
        for r in kept:
            by[r["split"]][r["ex"]].append(r)
        srcs = res.get("sources") or []
        src_ids = list(dict.fromkeys(s["id"] for s in srcs))
        lic = list(dict.fromkeys(s.get("license", "unknown") for s in srcs)) or ["unknown"]
        source, license_ = "+".join(src_ids), "+".join(map(str, lic))
        counts, golds, n_avail = {}, {}, {}
        out_path = os.path.join(a.out, f"{t}.jsonl")
        with open(out_path + ".tmp", "w") as f:
            for split in ("train", "calib", "test"):
                if split not in by:
                    continue
                cap = a.heldout_test_cap if (res["heldout"] and res["suite"] == "s1o" and split == "test") else caps[split]
                exids = sorted(by[split])
                stable_rng(a.seed, t, split, "cap").shuffle(exids)
                n_avail[split] = sum(len(v) for v in by[split].values())
                n, gc = 0, Counter()
                for e in exids:
                    for r in sorted(by[split][e], key=lambda x: x["q"]):
                        if n >= cap:
                            break
                        row = {"id": f"{t}/{split}/{n:06d}", "task": t, "family": res["family"], "split": split,
                               "heldout": res["heldout"], "state": r["state"], "question": r["question"],
                               "gold": r["gold"], "source": source, "license": license_}
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        gc[(r["question"]["type"], r["question"].get("instructions"), r["gold"])] += 1
                        n += 1
                    if n >= cap:
                        break
                counts[split] = n
                gq = Counter()
                for (_, _, g), c in gc.items():
                    gq[g] += c
                golds[split] = {"n_distinct_gold": len(gq),
                                "majority_gold": gq.most_common(1)[0][0] if gq else None,
                                "majority_share": round(gq.most_common(1)[0][1] / n, 3) if n else None}
        os.replace(out_path + ".tmp", out_path)
        qtypes = Counter(r["question"]["type"] for r in kept)
        manifest_tasks[t] = {
            "suite": res["suite"], "family": res["family"], "heldout": res["heldout"],
            "source": source, "license": license_, "sources": srcs,
            "question_types": dict(qtypes), "counts": counts, "available_before_cap": n_avail,
            "cap": {s: (a.heldout_test_cap if (res["heldout"] and res["suite"] == "s1o" and s == "test") else caps[s])
                    for s in counts},
            "gold": golds, "conversion": res.get("stats", {}),
            "train_rows_dropped_decontam": dict(contam),
            **({k: res[k] for k in ("note", "upstream_train_examples_discarded") if k in res}),
        }
        print(f"{t:24s} {res['family']:11s} {'HELDOUT' if res['heldout'] else '':7s} {counts} "
              f"decontam={sum(contam.values())}", flush=True)

    # which trained s1o tasks share a source dataset with each jev task (held-out-family caveat)
    trained_src = defaultdict(list)
    for t, m in manifest_tasks.items():
        if m["suite"] == "s1o" and not m["heldout"]:
            for s in m["sources"]:
                trained_src[s["id"]].append(t)
    for t, m in manifest_tasks.items():
        if m["suite"] == "jev":
            m["same_source_as_trained_s1o_tasks"] = sorted({x for s in m["sources"] for x in trained_src[s["id"]]})

    skipped = {t: {"reason": r.get("error", "unknown"), "family": r.get("family"), "heldout": r.get("heldout"),
                   "sources": r.get("sources")} for t, r in results.items() if not r.get("ok")}
    reduced = sorted(t for t, m in manifest_tasks.items() if m["conversion"].get("choice_reduced_to_20"))
    tot = Counter()
    for m in manifest_tasks.values():
        for s, c in m["counts"].items():
            tot[s] += c
    manifest = {
        "name": "decisions-v1",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "experiments/decisions/DATA-FORMAT.md",
        "builder": "experiments/decisions/build_dataset.py",
        "code_attribution": "s1o loaders adapted from mithalouni/system-one-open s1/data_real.py "
                            "(MIT, (c) 2026 Rami Mithalouni); each dataset keeps its own license",
        "config": {"seed": a.seed, "caps": {"train": a.train_cap, "calib": a.calib_cap, "test": a.test_cap,
                                            "heldout_test": a.heldout_test_cap},
                   "max_state_chars": a.max_state_chars, "max_choice_options": MAX_CHOICE,
                   "max_score_levels": MAX_SCORE,
                   "choice_over_20": "reduced per example to a deterministic 20-option subset (gold and any "
                                     "'none of the above' option always kept, dataset order preserved)",
                   "s1o_eval_split": "upstream eval examples shuffled (seeded) and halved into calib / test",
                   "s1o_heldout": "upstream heldout=True tasks: eval examples only, all split=test, heldout=true",
                   "jev_heldout_families": sorted(heldout_fams),
                   "decontamination": "train rows dropped if their format-normalised state matches any "
                                      "calib/test state of any task (200-char prefix, or exact if shorter); "
                                      "JSON-shaped states exempt"},
        "totals": {"tasks": len(manifest_tasks), "rows": dict(tot),
                   "state_truncated": sum(m["conversion"].get("state_truncated", 0) for m in manifest_tasks.values()),
                   "train_rows_dropped_decontam": sum(sum(m["train_rows_dropped_decontam"].values())
                                                      for m in manifest_tasks.values())},
        "choice_reduced_tasks": reduced,
        "skipped": skipped,
        "tasks": manifest_tasks,
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {len(manifest_tasks)} tasks, skipped {len(skipped)}; rows {dict(tot)}")
    return manifest


# ── validation ──────────────────────────────────────────────────────────────────────────────
def load_compile_question():
    """The shared prompt module (s1prompt.py) is what the server, trainer and evals render with."""
    sys.path.insert(0, HERE)
    from s1prompt import compile_question, MAX_CHOICE as mc, MAX_SCORE as ms
    assert (mc, ms) == (MAX_CHOICE, MAX_SCORE), "serving limits changed in s1prompt.py"
    return compile_question


def validate(a):
    compile_question = load_compile_question()
    man = json.load(open(os.path.join(a.out, "manifest.json")))
    errors, n_rows, ids = [], 0, set()
    for t, m in man["tasks"].items():
        if m["suite"] == "s1o":
            expect = {"test"} if m["heldout"] else {"train", "calib", "test"}
        else:
            expect = {"calib", "test"} if m["heldout"] else {"train", "calib", "test"}
        seen = Counter()
        for ln, line in enumerate(open(os.path.join(a.out, f"{t}.jsonl"))):
            try:
                r = json.loads(line)
            except Exception as e:
                errors.append(f"{t}:{ln} unparseable {e}")
                continue
            n_rows += 1
            try:
                assert set(r) == {"id", "task", "family", "split", "heldout", "state", "question", "gold",
                                  "source", "license"}, sorted(r)
                assert r["id"] not in ids, "dup id"
                ids.add(r["id"])
                assert r["task"] == t and r["family"] == m["family"] and r["heldout"] == m["heldout"]
                assert r["split"] in expect, f"unexpected split {r['split']}"
                assert isinstance(r["state"], str) and r["state"].strip(), "empty state"
                assert len(r["state"]) <= man["config"]["max_state_chars"], "state too long"
                q = r["question"]
                assert q["type"] in ("noul", "choice", "score")
                assert isinstance(q.get("instructions"), str) and q["instructions"].strip()
                if q["type"] == "noul":
                    c = q.get("criteria")
                    assert c is None or set(c) <= {"true", "false"}
                elif q["type"] == "choice":
                    assert isinstance(q["criteria"], dict) and 2 <= len(q["criteria"]) <= MAX_CHOICE
                    assert all(isinstance(k, str) and k for k in q["criteria"])
                else:
                    assert isinstance(q["criteria"], list) and 1 <= len(q["criteria"]) <= MAX_SCORE
                block, labels, outcomes = compile_question(q)
                assert r["gold"] in outcomes, f"gold {r['gold']!r} not in outcomes"
                assert len(labels) == len(outcomes) and len(set(outcomes)) == len(outcomes)
                seen[r["split"]] += 1
            except AssertionError as e:
                errors.append(f"{t}:{ln} {e}")
        if set(seen) != expect:
            errors.append(f"{t}: splits {sorted(seen)} != expected {sorted(expect)}")
        if dict(seen) != {k: v for k, v in m["counts"].items() if v}:
            errors.append(f"{t}: counts {dict(seen)} != manifest {m['counts']}")
    print(f"validated {n_rows} rows across {len(man['tasks'])} tasks: {len(errors)} errors")
    for e in errors[:50]:
        print("  ", e)
    return not errors


# ── main ────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--parts-dir", default="/mnt/data/datasets/hf-cache/decisions-v1-parts",
                    help="raw per-task rows before decontam/caps (a cache; safe to delete)")
    ap.add_argument("--train-cap", type=int, default=12000)
    ap.add_argument("--calib-cap", type=int, default=300)
    ap.add_argument("--test-cap", type=int, default=300)
    ap.add_argument("--heldout-test-cap", type=int, default=None,
                    help="test cap for s1o held-out tasks (default: --test-cap)")
    ap.add_argument("--max-state-chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jev-heldout-families", default=DEFAULT_JEV_HELDOUT)
    ap.add_argument("--tasks", nargs="*", help="only these s1o tasks (default: all 92)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per s1o task")
    ap.add_argument("--force", action="store_true", help="rebuild parts that already exist")
    ap.add_argument("--finalize-only", action="store_true", help="reuse existing parts")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.heldout_test_cap is None:
        a.heldout_test_cap = a.test_cap
    os.makedirs(a.parts_dir, exist_ok=True)
    for d in (os.environ["HF_DATASETS_CACHE"], os.environ["HF_HUB_CACHE"]):
        os.makedirs(d, exist_ok=True)

    if a.worker:
        sys.exit(run_worker(a.worker, a))
    if a.validate_only:
        sys.exit(0 if validate(a) else 1)

    names = a.tasks or list(TASKS)
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        sys.exit(f"unknown tasks: {unknown}")
    if not a.finalize_only:
        todo = [n for n in names if a.force or not os.path.exists(os.path.join(a.parts_dir, f"{n}.result.json"))]
        print(f"s1o: {len(todo)} tasks to build ({len(names) - len(todo)} cached), jobs={a.jobs}", flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def one(n):
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", n, "--parts-dir", a.parts_dir,
                   "--train-cap", str(a.train_cap), "--calib-cap", str(a.calib_cap), "--test-cap", str(a.test_cap),
                   "--heldout-test-cap", str(a.heldout_test_cap), "--max-state-chars", str(a.max_state_chars),
                   "--seed", str(a.seed)]
            rp = os.path.join(a.parts_dir, f"{n}.result.json")
            if os.path.exists(rp):
                os.remove(rp)
            t0 = time.time()
            with open(os.path.join(a.parts_dir, f"{n}.log"), "w") as log:
                try:
                    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            if not os.path.exists(rp):
                tail = open(os.path.join(a.parts_dir, f"{n}.log")).read()[-300:].replace("\n", " | ")
                why = f"timed out after {a.timeout}s" if rc == "timeout" else f"worker crashed rc={rc}: {tail}"
                json.dump({"task": n, "suite": "s1o", "family": TASKS[n]["group"], "heldout": TASKS[n]["heldout"],
                           "ok": False, "error": why}, open(rp, "w"), indent=1)
            r = json.load(open(rp))
            print(f"  {n:24s} {'ok' if r.get('ok') else 'SKIP':5s} {time.time() - t0:6.0f}s "
                  f"{'' if r.get('ok') else r.get('error', '')[:120]}", flush=True)

        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(one, todo))

    heldout_fams = {f.strip() for f in a.jev_heldout_families.split(",") if f.strip()}
    jev = build_jev(a, heldout_fams)
    results = {}
    for n in list(TASKS) + jev:  # manifest always covers every built task, even with --tasks
        rp = os.path.join(a.parts_dir, f"{n}.result.json")
        if os.path.exists(rp):
            results[n] = json.load(open(rp))
    finalize(a, results, heldout_fams)
    ok = validate(a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
