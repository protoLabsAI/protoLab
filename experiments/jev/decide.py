#!/usr/bin/env python3
"""One-forward-pass decisions: prompt with letter-labelled options, read the next-token
distribution over the label letters, renormalise. Same prompt and same readout for every arm.

  --arm smart                 our Qwen3.8-27B-NVFP4 lane (:8041), raw top_logprobs (cap 20)
  --arm hf:Qwen/Qwen3.5-2B    local transformers on CPU, exact logits (never touches prod GPUs)

Writes results/<arm>/<suite>.<split>.jsonl : {id, y, probs[], label_mass}
label_mass = raw probability the model put on ANY valid label before renormalising — a direct
read of how "in-schema" the model is without a grammar.
"""
import argparse, json, math, os, sys, time, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["injection", "sentiment", "routing", "langid"]
MAX_CHARS = 1500

def display(meta, o):
    # routing categories are ALL-CAPS in the source; lowercase makes every option ONE token
    return o.lower() if meta["suite"] == "routing" else o

def prompt(meta, text, fmt):
    text = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " [...]"
    if fmt == "letter":
        opts = "\n".join(f"{chr(65+i)}) {display(meta, o)}" for i, o in enumerate(meta["options"]))
        ask = "Answer with the letter of the correct option only."
    else:
        opts = "\n".join(f"- {display(meta, o)}" for o in meta["options"])
        ask = "Answer with the correct option exactly as written, and nothing else."
    return (f"{meta['question']}\n\nOptions:\n{opts}\n\nText:\n\"\"\"\n{text}\n\"\"\"\n\n{ask}")

def label_ids(tok, meta, fmt):
    """Token id each option must START with. Unique per suite (verified), so under a grammar that
    only permits the option strings, P(first token) == P(option) exactly."""
    strs = [chr(65+i) for i in range(len(meta["options"]))] if fmt == "letter" \
           else [display(meta, o) for o in meta["options"]]
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in strs]
    assert len(set(ids)) == len(ids), f"first-token collision in {meta['suite']}/{fmt}"
    return ids

def load(suite, split):
    meta = json.load(open(f"{HERE}/data/{suite}.meta.json"))
    rows = [json.loads(l) for l in open(f"{HERE}/data/{suite}.{split}.jsonl")]
    return meta, rows

# ---------- vLLM lane ----------
def smart_one(url, model, meta, row, fmt, ids):
    import urllib.request
    body = {"model": model, "messages": [{"role": "user", "content": prompt(meta, row["text"], fmt)}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}), timeout=120)
            d = json.load(r); break
        except Exception as e:
            if attempt == 3: raise
            time.sleep(2 * (attempt + 1))
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    got = {int(t["token"].split(":", 1)[1]): math.exp(t["logprob"]) for t in top}
    p = [got.get(i) for i in ids]
    mass = sum(x for x in p if x is not None)
    # options outside the top-20 share the leftover mass equally (tiny in practice; counted)
    missing = sum(x is None for x in p)
    left = max(0.0, 1.0 - sum(got.values()))
    p = [x if x is not None else left / max(1, missing) + 1e-12 for x in p]
    z = sum(p); return [x / z for x in p], mass, missing

# ---------- local transformers on CPU ----------
class HF:
    def __init__(self, name, threads):
        import torch
        torch.set_num_threads(threads)
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
        m.chunk_gated_delta_rule = m.fused_recurrent_gated_delta_rule = m.FusedRMSNormGated = None
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cpu").eval()
    def one(self, meta, row, fmt, ids):
        s = self.tok.apply_chat_template([{"role": "user", "content": prompt(meta, row["text"], fmt)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = self.tok(s, return_tensors="pt")          # batch 1: no padding through the recurrent layers
        with self.torch.no_grad():
            lg = self.model(**x).logits[0, -1].float()
        sel = self.torch.tensor(ids)
        full = self.torch.softmax(lg, -1)
        return self.torch.softmax(lg[sel], -1).tolist(), full[sel].sum().item(), 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--splits", default="test,calib")
    ap.add_argument("--url", default="http://localhost:8041/v1/chat/completions")
    ap.add_argument("--model", default="smart")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--format", choices=["letter", "name"], default="name")
    a = ap.parse_args()
    tag = a.arm.replace("hf:", "").split("/")[-1] if a.arm.startswith("hf:") else a.arm
    tag = f"{tag}-{a.format}"
    out_dir = f"{HERE}/results/{tag}"; os.makedirs(out_dir, exist_ok=True)
    hf = HF(a.arm[3:], a.threads) if a.arm.startswith("hf:") else None
    if hf: tok = hf.tok
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")   # same vocab as the served NVFP4
    for suite in a.suites.split(","):
        for split in a.splits.split(","):
            meta, rows = load(suite, split)
            ids = label_ids(tok, meta, a.format)
            if a.limit: rows = rows[:a.limit]
            path = f"{out_dir}/{suite}.{split}.jsonl"
            t0 = time.time(); res = [None] * len(rows)
            def job(i):
                r = rows[i]
                probs, mass, miss = hf.one(meta, r, a.format, ids) if hf else smart_one(a.url, a.model, meta, r, a.format, ids)
                return i, {"id": r["id"], "y": meta["options"].index(r["label"]), "probs": probs,
                           "label_mass": mass, "missing": miss}
            if hf:
                for i in range(len(rows)): res[i] = job(i)[1]
            else:
                with cf.ThreadPoolExecutor(a.workers) as ex:
                    for i, rec in ex.map(job, range(len(rows))): res[i] = rec
            with open(path, "w") as f:
                for rec in res: f.write(json.dumps(rec) + "\n")
            acc = sum(max(range(len(r["probs"])), key=lambda j: r["probs"][j]) == r["y"] for r in res) / len(res)
            dt = time.time() - t0
            print(f"{tag:20s} {suite:10s} {split:6s} n={len(res):4d} acc={acc:.3f} "
                  f"label_mass={sum(r['label_mass'] for r in res)/len(res):.3f} missing={sum(r['missing'] for r in res)} {dt:6.1f}s ({1000*dt/len(res):.0f}ms/item)", flush=True)

if __name__ == "__main__":
    main()
