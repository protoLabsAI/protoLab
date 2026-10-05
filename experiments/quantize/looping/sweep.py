"""Ornith-1.5-9B looping sweep: sampling arms x prompts against a live llama-server.

Usage: python3 sweep.py <port> <rung-label> <out.jsonl> [arms...]
"""
import json
import sys
import time

import requests

from loopdet import analyse

URL = "http://127.0.0.1:{port}/v1/chat/completions"

# ---- sampling arms -------------------------------------------------------
# llamacpp_default is what a user gets if they run our card's command line
# verbatim: we ship no sampler flags, so llama.cpp's own defaults apply.
# upstream_* are what ornith-ai's model card recommends.
ARMS = {
    "llamacpp_default": dict(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05,
                             presence_penalty=0.0),
    "upstream_general": dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                             presence_penalty=1.5),
    "upstream_coding": dict(temperature=0.6, top_k=20, top_p=0.95, min_p=0.0,
                            presence_penalty=0.0),
    "greedy": dict(temperature=0.0, top_k=1, top_p=1.0, min_p=0.0,
                   presence_penalty=0.0),
    # isolates presence_penalty from temperature: default arm + pp only
    "default_plus_pp15": dict(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05,
                              presence_penalty=1.5),
    # isolates temperature from pp: upstream temp/topk, but no pp
    "upstream_nopp": dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                          presence_penalty=0.0),
    # the classic llama.cpp anti-loop sampler, for comparison
    "default_plus_dry": dict(temperature=0.8, top_k=40, top_p=0.95, min_p=0.05,
                             presence_penalty=0.0, dry_multiplier=0.8,
                             dry_base=1.75, dry_allowed_length=2),
}

# ---- prompts: long-form / enumerative / open-ended, the loop-prone shapes --
PROMPTS = [
    ("story", "Write a 1500-word short story about a lighthouse keeper who "
              "discovers the light has been going out on its own."),
    ("enumerate", "List 60 distinct, specific uses for a shipping container. "
                  "Number them. Do not repeat an idea."),
    ("prose", "Write a detailed, atmospheric description of an abandoned "
              "railway station across four seasons. At least 1000 words."),
    ("chat", "I've been feeling stuck at work for months. Talk it through with "
             "me properly - ask me questions, push back, don't just list advice."),
    ("explain", "Explain how a modern CPU branch predictor works, in depth, "
                "including history, TAGE, and why mispredicts are expensive."),
    ("scaffold", "Write a travel guide covering all 12 months in Kyoto. For "
                 "each month give weather, festivals, food, and one hidden spot."),
    ("code", "Write a complete Python implementation of a red-black tree with "
             "insert, delete, search, and an in-order iterator. Include tests."),
    ("roleplay", "You are a grizzled ship's quartermaster in 1720. I'm a new "
                 "recruit asking about the articles. Stay in character."),
]


def run(port, arm_name, params, pid, prompt, max_tokens=1500, thinking=False):
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": thinking},
        **params,
    }
    t0 = time.time()
    r = requests.post(URL.format(port=port), json=body, timeout=1200)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    a = analyse(content)
    return {
        "arm": arm_name, "prompt_id": pid, "thinking": thinking,
        "finish_reason": ch.get("finish_reason"),
        "content_chars": len(content), "reasoning_chars": len(reasoning),
        "completion_tokens": j.get("usage", {}).get("completion_tokens"),
        "elapsed_s": round(time.time() - t0, 1),
        **{k: v for k, v in a.items() if k != "chars"},
        "content": content,
    }


def main():
    import os
    port, rung, out = sys.argv[1], sys.argv[2], sys.argv[3]
    arms = sys.argv[4:] or list(ARMS)
    think = os.environ.get("THINK") == "1"
    mt = int(os.environ.get("MAXTOK", "1500"))
    n = 0
    with open(out, "a") as f:
        for arm in arms:
            for pid, prompt in PROMPTS:
                try:
                    rec = run(port, arm, ARMS[arm], pid, prompt, max_tokens=mt, thinking=think)
                except Exception as e:  # noqa: BLE001
                    rec = {"arm": arm, "prompt_id": pid, "error": repr(e)}
                rec["rung"] = rung
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                flag = "LOOP" if rec.get("looped") else "ok  "
                print(f"{rung:9s} {arm:18s} {pid:10s} {flag} "
                      f"rep15={rec.get('rep15')} "
                      f"cyc={(rec.get('tail_cycle') or {}).get('period')} "
                      f"fin={rec.get('finish_reason')} "
                      f"tok={rec.get('completion_tokens')} "
                      f"{rec.get('elapsed_s')}s", flush=True)
    print(f"done: {n} samples -> {out}")


if __name__ == "__main__":
    main()
