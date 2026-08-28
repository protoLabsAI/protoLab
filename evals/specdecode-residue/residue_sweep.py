"""Sweep every prompt-length residue mod 128 against a warm lane, scoring verbatim copy.

Tests "Bug B" from syv-ai/qwen38-27b-rtx3090 (docs/gotchas.md #37) on our stack:
under a CAPTURED (FULL) verify step, a speculative request that HITS the prefix cache
and whose prompt length lands on one particular residue mod 128 collapses. Their fit is
R = 117 + k, so k=3 predicts residue 120. Their observation was on vLLM 0.27.1 / RTX
3090 / lossy KVarN KV; this script asks whether it reproduces on 0.25.1 / sm120 / NVFP4.

Two conditions, both required and both reproduced here:
  hit      each prompt is sent TWICE; the second (a full self-hit) is the measurement.
  residue  the pad steps by exactly one token, so 128 samples cover all 128 residues
           exactly once. Sampling 5 residues misses a single break 96% of the time --
           sweep all of them or say you sampled.

  python evals/specdecode-residue/residue_sweep.py <label> [--port N] [--start N] [--count N]
"""
import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verbatim import classify, coverage, median, prefix_match  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--port", type=int, default=8042)
ap.add_argument("--model", default="smart")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=128)
ap.add_argument("--corpus", default=os.environ.get("CORPUS", ""))
ap.add_argument("--tokenizer", default="/mnt/models/quantized/Qwen3.8-27B-NVFP4")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--out", default="")
A = ap.parse_args()

BASE = f"http://127.0.0.1:{A.port}"
DOC = open(A.corpus).read()
LINES = "\n".join(DOC.split("\n")[:40])          # what we ask it to reproduce
INSTR = ("\n\nReproduce the first 40 lines of the document above, exactly as written, "
         "verbatim and with nothing else added.")
PAD = " x"                                        # exactly one token (id 830)

from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained(A.tokenizer)


def plen(content):
    """Token length of the full rendered chat prompt, which is what the residue is of."""
    s = TOK.apply_chat_template([{"role": "user", "content": content}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(x) for x in re.findall(
        rf"^{re.escape(n)}\{{[^}}]*\}} ([0-9.e+-]+)$", t, re.M))
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")


def once(content):
    body = json.dumps({
        "model": A.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": A.max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    d0 = metrics()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=1800).read().decode())
    d1 = metrics()
    ch = r["choices"][0]
    msg = ch["message"]
    # `or ""`: a collapse-to-stop returns content=null, and indexing that raises
    # TypeError instead of producing a finding.
    ans = msg.get("content") or ""
    rea = msg.get("reasoning_content") or msg.get("reasoning") or ""
    dn = d1[0] - d0[0]
    u = r.get("usage", {}) or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "ans": ans, "reasoning_chars": len(rea),
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached,
        "tok_per_step": ((d1[1] - d0[1]) / dn + 1) if dn else float("nan"),
    }


rows = []
t0 = time.time()
for i in range(A.start, A.start + A.count):
    content = DOC + INSTR + PAD * i
    n = plen(content)
    once(content)                     # warm: populate the prefix cache
    m = once(content)                 # measure: a full self-hit
    flag, cov, why = classify(m["ans"], DOC,
                              ref=median([r["cov"] for r in rows[-16:]]) if len(rows) >= 8 else None)
    rows.append({"pad": i, "prompt_tokens": n, "residue": n % 128, "cov": cov,
                 "prefix": prefix_match(m["ans"], DOC), "chars": len(m["ans"]),
                 "flag": flag, "why": why, **{k: m[k] for k in
                 ("finish_reason", "reasoning_chars", "cached_tokens", "tok_per_step")}})
    r = rows[-1]
    mark = "  <-- BROKEN" if flag == "BROKEN" else ""
    print(f"pad={i:3d} len={n:6d} r={r['residue']:3d} cov={cov:.2f} chars={r['chars']:4d} "
          f"cached={r['cached_tokens']:6d} tok/step={r['tok_per_step']:.2f} "
          f"fin={r['finish_reason']}{mark} {why}", flush=True)

broken = [r for r in rows if r["flag"] == "BROKEN"]
covs = [r["cov"] for r in rows]
print(f"\n{A.label}: {len(rows)} residues in {time.time()-t0:.0f}s, "
      f"median coverage {median(covs):.3f}, BROKEN {len(broken)}")
for r in broken:
    print(f"  residue {r['residue']:3d} (len {r['prompt_tokens']}): {r['why']}")

out = A.out or f"evals/results/specdecode-residue/{A.label}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"label": A.label, "port": A.port, "rows": rows}, open(out, "w"), indent=1)
print("wrote", out)
