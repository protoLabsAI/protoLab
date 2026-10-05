#!/usr/bin/env python3
"""Grade (or re-grade) an effort-sweep run from its raw.jsonl. Generation is the expensive
half — this lets a killed/timed-out sweep be finished, or graders be changed, for free."""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sweep import grade, lcb_items, reasoning_hard_items

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("--lcb-limit", type=int, default=None, help="default: from manifest.json")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--max-tests", type=int, default=20)
ap.add_argument("--per-test-timeout", type=int, default=6)
ap.add_argument("--problem-budget", type=float, default=90.0)
a = ap.parse_args()

rd = Path(a.rundir)
man = json.loads((rd / "manifest.json").read_text())
lim = a.lcb_limit or man.get("lcb_limit", 20)

index = {}
for it in (lcb_items(lim, "hard")
           + lcb_items(man.get("lcb_medium_limit", 0) or 0, "medium")
           + reasoning_hard_items()):
    index[(it["suite"], it["id"])] = it

raw = [json.loads(l) for l in open(rd / "raw.jsonl")]
print(f"{len(raw)} generations, {len(index)} known items -> grading with {a.workers} workers")

def one(rec):
    it = index.get((rec["suite"], rec["id"]))
    if it is None:
        return {"score": 0.0, "grade_note": "item_not_found"}
    return grade(it, rec, a.max_tests, a.per_test_timeout, a.problem_budget)

out = []
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    futs = {ex.submit(one, r): r for r in raw}
    for i, f in enumerate(as_completed(futs), 1):
        r = futs[f]
        try:
            g = f.result()
        except Exception as e:
            g = {"score": 0.0, "grade_note": f"grader_error: {e}"[:200]}
        out.append({"suite": r["suite"], "id": r["id"], "effort": r["effort"],
                    "trial": r["trial"], "score": g["score"],
                    "passed": g.get("passed"), "total": g.get("total"),
                    "grade_note": g.get("grade_note", ""),
                    "reasoning_tokens": r.get("reasoning_tokens"),
                    "answer_tokens": r.get("answer_tokens"),
                    "completion_tokens": r.get("completion_tokens"),
                    "latency_s": r.get("latency_s"),
                    "finish_reason": r.get("finish_reason"),
                    "error": r.get("error")})
        if i % 50 == 0:
            print(f"  graded {i}/{len(raw)}", flush=True)

(rd / "results.json").write_text(json.dumps(out, indent=2))
print(f"wrote {rd/'results.json'} ({len(out)} rows)")
