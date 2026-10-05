"""Long generation that overruns a small context, to expose context-shift damage.

Prompt is ~1200 tokens of real prose; we then ask for up to 6000 more tokens.
With ctx 4096 that guarantees the window fills partway through generation.
"""
import json
import sys
import time

import requests

from loopdet import analyse

URL = "http://127.0.0.1:{port}/v1/chat/completions"
CALIB = "/mnt/data/gguf-forge/Ornith-1.5-9B-MTP/imatrix/calibration.txt"

TASKS = [
    ("essay", "Write a very long, continuous analytical essay (at least 4000 "
              "words) about the document above: its themes, its structure, its "
              "assumptions, and what it leaves out. Do not use bullet points."),
    ("story", "Using the document above only as tonal inspiration, write a very "
              "long continuous short story, at least 4000 words, about a night "
              "shift at a remote weather station. Do not stop early."),
    ("manual", "Write an exhaustive technical manual, at least 4000 words, for "
               "maintaining a mechanical telegraph network. Cover parts, "
               "faults, schedules, and troubleshooting in depth."),
]


def main():
    port, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(CALIB, encoding="utf-8", errors="ignore") as f:
        filler = f.read()[:4400]          # ~1200 tokens
    cut = filler.rfind(". ")
    filler = filler[:cut + 1]

    with open(out, "a") as f:
        for tid, task in TASKS:
            body = {
                "model": "local",
                "messages": [{"role": "user",
                              "content": filler + "\n\n---\n\n" + task}],
                "max_tokens": 6000,
                "seed": 1234,
                "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            t0 = time.time()
            try:
                r = requests.post(URL.format(port=port), json=body, timeout=3600)
                r.raise_for_status()
                j = r.json()
                ch = j["choices"][0]
                content = ch["message"].get("content") or ""
                a = analyse(content)
                u = j.get("usage", {})
                rec = {
                    "arm": label, "task": tid,
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "finish_reason": ch.get("finish_reason"),
                    "elapsed_s": round(time.time() - t0, 1),
                    **{k: v for k, v in a.items() if k != "chars"},
                    "content_tail": content[-800:],
                }
            except Exception as e:  # noqa: BLE001
                rec = {"arm": label, "task": tid, "error": repr(e)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            flag = "LOOP" if rec.get("looped") else "ok  "
            print(f"{label:10s} {tid:8s} {flag} "
                  f"ptok={rec.get('prompt_tokens')} "
                  f"ctok={rec.get('completion_tokens')} "
                  f"rep15={rec.get('rep15')} "
                  f"cyc={(rec.get('tail_cycle') or {}).get('period')} "
                  f"fin={rec.get('finish_reason')} "
                  f"{rec.get('elapsed_s')}s", flush=True)


if __name__ == "__main__":
    main()
