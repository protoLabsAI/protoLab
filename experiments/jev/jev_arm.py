#!/usr/bin/env python3
"""Jev reference arm — TypeSafe /v1/systemone, one `choice` question per item.

Fairness: Jev gets the same information the local arms got — state = the text (same 1500-char
cap), instructions = the same question, criteria = the same option names with NO extra rubric
(null), exactly as the local arms saw a bare option list. Writes the same result format as
decide.py, so metrics.py / bootstrap.py score it with no changes.

Key: TYPESAFE_API_KEY from the environment or experiments/jev/.env (gitignored). Never logged.
  --dry-run   print one request body + a token/cost estimate, make NO network call.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import decide as D
URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")
PRICE_PER_MTOK = 0.042          # input only; output free (docs.typesafe.ai/models)

def key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if not k and os.path.exists(f"{HERE}/.env"):
        for line in open(f"{HERE}/.env"):
            if line.startswith("TYPESAFE_API_KEY="): k = line.split("=", 1)[1].strip()
    return k

def body(meta, row, model):
    text = row["text"] if len(row["text"]) <= D.MAX_CHARS else row["text"][:D.MAX_CHARS] + " [...]"
    return {"state": text, "model": model,
            "questions": {"q": {"type": "choice", "instructions": meta["question"],
                                "criteria": {D.display(meta, o): None for o in meta["options"]}}}}

def call(b, k):
    for attempt in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(b).encode(),
                    headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"}), timeout=60)
            return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 5: time.sleep(2 ** attempt); continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(D.SUITES)); ap.add_argument("--splits", default="test")
    ap.add_argument("--model", default="jev-1.13.0"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")      # token ESTIMATE only; Jev's tokenizer is undisclosed
    plan, est = [], 0
    for s in a.suites.split(","):
        for sp in a.splits.split(","):
            meta, rows = D.load(s, sp); rows = rows[:a.limit] if a.limit else rows
            for r in rows:
                b = body(meta, r, a.model); est += len(tok.encode(json.dumps(b))); plan.append((s, sp, meta, r, b))
    print(f"plan: {len(plan)} calls, ~{est:,} input tokens (Qwen-tokenizer estimate) "
          f"=> ~${est/1e6*PRICE_PER_MTOK:.4f} at ${PRICE_PER_MTOK}/MTok", flush=True)
    if a.dry_run:
        print("sample request body:\n" + json.dumps(plan[0][4], indent=1)[:900]); return
    k = key()
    if not k: sys.exit("no TYPESAFE_API_KEY in env or experiments/jev/.env — refusing to run")
    out_dir = f"{HERE}/results/jev-{a.model}"; os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for p in plan: groups.setdefault((p[0], p[1]), []).append(p)
    used = 0
    for (s, sp), items in groups.items():
        t0 = time.time()
        def job(p):
            _, _, meta, r, b = p; d = call(b, k); ans = d["answers"]["q"]
            opts = [D.display(meta, o) for o in meta["options"]]
            probs = [float(ans["probabilities"].get(o, 0.0)) for o in opts]
            z = sum(probs) or 1.0
            return {"id": r["id"], "y": meta["options"].index(r["label"]), "probs": [x / z for x in probs],
                    "label_mass": z, "missing": sum(o not in ans["probabilities"] for o in opts),
                    "jev_choice": ans.get("choice"), "jev_confidence": ans.get("confidence"),
                    "usage": d.get("usage", {})}, 
        with cf.ThreadPoolExecutor(a.workers) as ex: res = [x[0] for x in ex.map(job, items)]
        with open(f"{out_dir}/{s}.{sp}.jsonl", "w") as f:
            for rec in res: f.write(json.dumps(rec) + "\n")
        u = sum(r["usage"].get("input_tokens", 0) for r in res); used += u
        acc = sum(max(range(len(r["probs"])), key=lambda j: r["probs"][j]) == r["y"] for r in res) / len(res)
        dt = time.time() - t0
        print(f"jev {s:10s} {sp:6s} n={len(res):4d} acc={acc:.3f} in_tok={u:,} {1000*dt/len(res):.0f}ms/item(wall,x{a.workers})", flush=True)
    print(f"billed input tokens: {used:,} => ${used/1e6*PRICE_PER_MTOK:.4f}")

if __name__ == "__main__":
    main()
