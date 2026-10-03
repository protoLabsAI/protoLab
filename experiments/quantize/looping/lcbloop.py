"""Reproduce the LiveCodeBench budget-exhaustion cases and KEEP the text.

Our 2026-08-22 scorecard measured 13/30 hard LCB problems consuming the entire
32,768-token budget with thinking OFF. The runner does not persist generations,
so we never established WHY. This replays the exact capping problems against a
local GGUF server with the text retained, then runs the degeneration detector.

That distinguishes the two hypotheses that look identical to a user:
  (a) genuine repetition loop  -> tail_cycle / high rep15
  (b) long non-repeating rambling that simply never emits a stop token
"""
import json
import sys
import time

import requests

sys.path.insert(0, "/home/ava/dev/lab/evals/runners")
sys.path.insert(0, "/tmp/claude-1001/-home-ava-dev-lab/60697cca-4e04-40f9-a218-548c0e1c5fb5/scratchpad")

from loopdet import analyse  # noqa: E402
from run_livecodebench import build_prompt, load_problems  # noqa: E402

URL = "http://127.0.0.1:{port}/v1/chat/completions"

# the 13 that hit the 32,768 cap on the 2026-08-22 NVFP4 scorecard
CAPPED = {"abc387_f", "3562", "abc388_e", "abc388_f", "abc388_g", "arc190_a",
          "arc190_c", "abc389_f", "abc389_g", "abc390_e", "abc390_f",
          "arc191_d", "abc392_g"}


def main():
    port, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n_max = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    budget = int(sys.argv[5]) if len(sys.argv) > 5 else 32768

    probs = load_problems("release_v6", "2025-01-01", ["hard"], 30)
    sel = [p for p in probs if p["question_id"] in CAPPED][:n_max]
    print(f"replaying {len(sel)} previously-capped problems, budget {budget}",
          flush=True)

    with open(out, "a") as f:
        for p in sel:
            body = {
                "model": "local",
                "messages": [{"role": "user", "content": build_prompt(p)}],
                "max_tokens": budget,
                "seed": 1234,
                "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            t0 = time.time()
            try:
                r = requests.post(URL.format(port=port), json=body, timeout=7200)
                r.raise_for_status()
                j = r.json()
                ch = j["choices"][0]
                content = ch["message"].get("content") or ""
                a = analyse(content)
                u = j.get("usage", {})
                rec = {
                    "arm": label, "qid": p["question_id"], "title": p["title"],
                    "completion_tokens": u.get("completion_tokens"),
                    "finish_reason": ch.get("finish_reason"),
                    "elapsed_s": round(time.time() - t0, 1),
                    **{k: v for k, v in a.items() if k != "chars"},
                    "content_head": content[:400],
                    "content_tail": content[-1500:],
                }
            except Exception as e:  # noqa: BLE001
                rec = {"arm": label, "qid": p["question_id"], "error": repr(e)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            flag = "LOOP" if rec.get("looped") else "ok  "
            print(f"{label:10s} {rec.get('qid','?'):10s} {flag} "
                  f"ctok={rec.get('completion_tokens')} "
                  f"rep15={rec.get('rep15')} "
                  f"cyc={(rec.get('tail_cycle') or {}).get('period')} "
                  f"blk={(rec.get('max_block') or {}).get('reps')} "
                  f"fin={rec.get('finish_reason')} "
                  f"{rec.get('elapsed_s')}s", flush=True)


if __name__ == "__main__":
    main()
