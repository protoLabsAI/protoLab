"""Does Ornith-1.5-9B loop as a function of CONTEXT DEPTH?

Upstream's most specific claim (deepreinforce-ai/Ornith-1.0-35B-GGUF #26) is that
past ~22k context the model degenerates into recursive generation. The 1.0-9B
reports are all agentic harnesses (Continue, OpenCode) i.e. long accumulated
context. Short single-turn prompts do NOT reproduce it -- so sweep depth.

Filler is real prose from the imatrix calibration corpus, not repeated text,
so the context itself cannot seed a repetition.
"""
import json
import sys
import time

import requests

from loopdet import analyse
from sweep import ARMS

URL = "http://127.0.0.1:{port}/v1/chat/completions"
CALIB = "/mnt/data/gguf-forge/Ornith-1.5-9B-MTP/imatrix/calibration.txt"

TASK = ("\n\n---\n\nYou have read the document above. Now write a thorough, "
        "well-organised analytical essay of at least 1200 words about its "
        "themes, structure and style. Do not summarise section by section; "
        "build an argument.")


def load_filler():
    with open(CALIB, encoding="utf-8", errors="ignore") as f:
        return f.read()


def make_prompt(filler, approx_tokens):
    # ~3.6 chars/token for English prose; trimmed to a sentence boundary.
    want = int(approx_tokens * 3.6)
    chunk = filler[:want]
    cut = chunk.rfind(". ")
    if cut > want * 0.8:
        chunk = chunk[:cut + 1]
    return "Read the following document.\n\n" + chunk + TASK


def run(port, arm, params, depth, prompt, max_tokens=2000):
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": False},
        **params,
    }
    t0 = time.time()
    r = requests.post(URL.format(port=port), json=body, timeout=2400)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    content = ch["message"].get("content") or ""
    a = analyse(content)
    u = j.get("usage", {})
    return {
        "arm": arm, "depth_target": depth,
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish_reason": ch.get("finish_reason"),
        "elapsed_s": round(time.time() - t0, 1),
        **{k: v for k, v in a.items() if k != "chars"},
        "content_tail": content[-600:],
    }


def main():
    port, rung, out = sys.argv[1], sys.argv[2], sys.argv[3]
    depths = [int(x) for x in sys.argv[4].split(",")]
    arms = sys.argv[5].split(",")
    filler = load_filler()
    with open(out, "a") as f:
        for depth in depths:
            prompt = make_prompt(filler, depth)
            for arm in arms:
                try:
                    rec = run(port, arm, ARMS[arm], depth, prompt)
                except Exception as e:  # noqa: BLE001
                    rec = {"arm": arm, "depth_target": depth, "error": repr(e)}
                rec["rung"] = rung
                f.write(json.dumps(rec) + "\n")
                f.flush()
                flag = "LOOP" if rec.get("looped") else "ok  "
                print(f"{rung:8s} d={depth:6d} {arm:18s} {flag} "
                      f"ptok={rec.get('prompt_tokens')} "
                      f"ctok={rec.get('completion_tokens')} "
                      f"rep15={rec.get('rep15')} "
                      f"cyc={(rec.get('tail_cycle') or {}).get('period')} "
                      f"fin={rec.get('finish_reason')} "
                      f"{rec.get('elapsed_s')}s", flush=True)


if __name__ == "__main__":
    main()
