"""Powered A/B at IQ2_M: does the model author's recommended sampling reduce looping?

The ladder found the rung effect at p=7e-7, but the sampling comparison inside
IQ2_M was 1/16 vs 8/32 -> Fisher p=0.24, i.e. underpowered and unquotable.
[[feedback_underpowered_is_not_null]] says report the required n and run it.

8 prompts x 8 seeds = 64 samples per arm.
"""
import json
import sys
import time

import requests

from loopdet import analyse
from sweep import ARMS, PROMPTS

URL = "http://127.0.0.1:{port}/v1/chat/completions"
SEEDS = [1234, 7, 99, 2026, 31337, 555, 8675309, 42]


def main():
    port, rung, out = sys.argv[1], sys.argv[2], sys.argv[3]
    arms = sys.argv[4].split(",")
    with open(out, "a") as f:
        for arm in arms:
            for seed in SEEDS:
                for pid, prompt in PROMPTS:
                    body = {
                        "model": "local",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1500, "seed": seed,
                        "chat_template_kwargs": {"enable_thinking": False},
                        **ARMS[arm],
                    }
                    t0 = time.time()
                    try:
                        r = requests.post(URL.format(port=port), json=body,
                                          timeout=1200)
                        r.raise_for_status()
                        j = r.json()
                        ch = j["choices"][0]
                        c = ch["message"].get("content") or ""
                        a = analyse(c)
                        rec = {"arm": arm, "seed": seed, "prompt_id": pid,
                               "rung": rung,
                               "finish_reason": ch.get("finish_reason"),
                               "completion_tokens":
                                   j.get("usage", {}).get("completion_tokens"),
                               "elapsed_s": round(time.time() - t0, 1),
                               **{k: v for k, v in a.items() if k != "chars"},
                               "tail": c[-300:]}
                    except Exception as e:  # noqa: BLE001
                        rec = {"arm": arm, "seed": seed, "prompt_id": pid,
                               "rung": rung, "error": repr(e)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{arm:18s} s={seed:<8d} {pid:10s} "
                          f"{'LOOP' if rec.get('looped') else 'ok  '} "
                          f"rep15={rec.get('rep15')}", flush=True)


if __name__ == "__main__":
    main()
