#!/usr/bin/env python3
"""Qwen3.8-27B reasoning_effort sweep — brevity vs quality.

The lane's DEFAULT is reasoning_effort=xhigh (see models/serve-qwen38-27b.sh). This measures
whether that default is the right operating point, across the only efforts the lane accepts:
  none (thinking off) | low | medium | xhigh (default)

Quality is JUDGE-FREE on both suites:
  lcb_hard        LiveCodeBench hard, execution-graded, partial credit (fraction of tests passed)
  reasoning_hard  solver-verified regex match on the required ANSWER line

Brevity is measured as reasoning tokens (tokenized on the lane's own /tokenize endpoint),
answer tokens, and total completion tokens. Latency is recorded but is UNDER CONCURRENT LOAD
(N workers against a load-balanced replica pair) — it is a secondary signal, tokens are primary.

Everything goes through the gateway (feedback_measure_through_the_gateway), which load-balances
:8041 and :8042. Sampling is held CONSTANT across arms so effort is the only variable.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LAB = Path("/home/ava/dev/lab")
sys.path.insert(0, str(LAB / "evals"))
sys.path.insert(0, str(LAB / "evals" / "runners"))

import urllib.request, urllib.error
import yaml
from runners.run_livecodebench import load_problems, build_prompt, grade_problem
from graders.match import MatchGrader

GATEWAY = os.environ.get("SWEEP_GATEWAY", "http://100.101.189.45:4000/v1")
MODEL   = os.environ.get("SWEEP_MODEL", "protolabs/smart")
KEY     = os.environ.get("LITELLM_API_KEY") or os.environ.get("GATEWAY_API_KEY", "not-needed")
TOKENIZE_URL = os.environ.get("SWEEP_TOKENIZE", "http://localhost:8041/tokenize")

# Held constant across every arm — Qwen3.8 thinking preset. Recorded in the manifest.
SAMPLING = {"temperature": 0.6, "top_p": 0.95}
SAMPLING_EXTRA = {"top_k": 20, "min_p": 0.0}
MAX_TOKENS = 32768          # prod budget (feedback_eval_prod_token_budget)

EFFORTS = ["none", "low", "medium", "xhigh"]

_print_lock = threading.Lock()
def log(*a):
    with _print_lock:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- generation
def ntok(text: str) -> int:
    """Exact token count via the lane's own tokenizer. 0 on failure (never fabricate)."""
    if not text:
        return 0
    try:
        req = urllib.request.Request(
            TOKENIZE_URL, data=json.dumps({"model": "smart", "prompt": text}).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60)).get("count", -1)
    except Exception:
        return -1


def call(prompt: str, effort: str) -> dict:
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": effort,
            **SAMPLING, **SAMPLING_EXTRA}
    req = urllib.request.Request(
        GATEWAY + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300], "latency_s": round(time.time() - t0, 2)}
    dt = time.time() - t0
    ch = r["choices"][0]
    m = ch["message"]
    psf = m.get("provider_specific_fields") or {}
    reasoning = (m.get("reasoning_content") or psf.get("reasoning")
                 or psf.get("reasoning_content") or "")
    content = m.get("content") or ""
    u = r.get("usage") or {}
    return {"content": content, "reasoning": reasoning,
            "finish_reason": ch.get("finish_reason"),
            "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"),
            "latency_s": round(dt, 2)}


# ----------------------------------------------------------------------------- task pools
def lcb_items(limit: int, difficulty: str = "hard") -> list[dict]:
    if not limit:
        return []
    probs = load_problems("release_v6", "2025-01-01", [difficulty], limit)
    return [{"suite": f"lcb_{difficulty}", "id": str(p["question_id"]),
             "prompt": build_prompt(p), "_problem": p} for p in probs]


def reasoning_hard_items() -> list[dict]:
    out = []
    for f in sorted((LAB / "evals" / "tasks" / "reasoning_hard").glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        for t in d["tests"]:
            out.append({"suite": "reasoning_hard", "id": t["id"], "prompt": t["prompt"],
                        "_graders": t["graders"]})
    return out


# ----------------------------------------------------------------------------- grading
def grade(item: dict, gen: dict, max_tests: int, per_test: int, budget: float) -> dict:
    if gen.get("error"):
        return {"score": 0.0, "grade_note": "model_error"}
    content = gen.get("content") or ""
    if item["suite"].startswith("lcb_"):
        # Match the LCB runner: fall back to the reasoning stream if content is empty
        # (budget burned mid-think -> usually no code fence -> 0, the correct outcome).
        text = content if content.strip() else gen.get("reasoning", "")
        g = grade_problem(item["_problem"], text, max_tests, per_test, budget)
        return {"score": g["score"], "passed": g["passed"], "total": g["total"],
                "grade_note": g.get("error", "")}
    # reasoning_hard: score with the repo's CANONICAL MatchGrader so these numbers stay
    # comparable to the rest of the board -- and so the LaTeX/markdown normalisation that has
    # produced false zeros before (reference_qwen38_effort_ceiling) is applied here too.
    # Graded on the FINAL content only: an answer that never leaves the think block is a miss.
    if "_graders" not in item:
        raise KeyError(f"no grader for suite {item['suite']!r} -- grading was mis-routed, "
                       f"which would silently score 0 for every run of that suite")
    scores, notes = [], []
    for gr in item["_graders"]:
        g = MatchGrader(dimension=gr.get("dimension", "match"),
                        mode=gr.get("mode", "contains"),
                        expected=gr.get("expected"),
                        case_sensitive=gr.get("case_sensitive", True))
        r = g.grade({}, {"output": content})
        scores.append(r.score)
        if r.score < 1.0:
            notes.append(f"{g.dimension}:{r.score:.2f}")
    n = len(scores)
    return {"score": sum(scores) / n if n else 0.0,
            "passed": round(sum(scores), 2), "total": n,
            "grade_note": ",".join(notes)[:80]}


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb-limit", type=int, default=20, help="LCB hard problems")
    ap.add_argument("--lcb-medium-limit", type=int, default=0, help="LCB medium problems")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--grade-workers", type=int, default=6)
    ap.add_argument("--efforts", default=",".join(EFFORTS))
    ap.add_argument("--suites", default="lcb_hard,reasoning_hard")
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--per-test-timeout", type=int, default=6)
    ap.add_argument("--problem-budget", type=float, default=90.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    efforts = args.efforts.split(",")
    suites = args.suites.split(",")
    items: list[dict] = []
    if "lcb_hard" in suites:
        items += lcb_items(args.lcb_limit, "hard")
    if "lcb_medium" in suites:
        items += lcb_items(args.lcb_medium_limit, "medium")
    if "reasoning_hard" in suites:
        items += reasoning_hard_items()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.out or (LAB / "experiments" / "effort-sweep" / f"run-{stamp}"))
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [(it, e, t) for it in items for e in efforts for t in range(args.trials)]
    # Shuffle (fixed seed) so the expensive arms do not all land in the first wave: that
    # clusters every long generation into one saturated batch, collapsing per-stream
    # throughput and making early partial results unrepresentative.
    random.Random(20260908).shuffle(jobs)
    log(f"items={len(items)} efforts={efforts} trials={args.trials} -> {len(jobs)} generations")
    log(f"out={outdir}")

    (outdir / "manifest.json").write_text(json.dumps({
        "model": MODEL, "gateway": GATEWAY, "sampling": {**SAMPLING, **SAMPLING_EXTRA},
        "max_tokens": MAX_TOKENS, "efforts": efforts, "trials": args.trials,
        "suites": suites, "lcb_limit": args.lcb_limit,
        "lcb_medium_limit": args.lcb_medium_limit, "client_timeout_s": 3600, "workers": args.workers,
        "n_items": len(items), "n_generations": len(jobs), "started": stamp,
        "note": "latency measured under concurrent load; tokens are the primary brevity metric",
    }, indent=2))

    # ---- phase 1: generate (network-bound, parallel)
    raw_path = outdir / "raw.jsonl"
    done = [0]
    t_start = time.time()
    lock = threading.Lock()
    fh = open(raw_path, "w")

    def work(job):
        it, effort, trial = job
        gen = call(it["prompt"], effort)
        rec = {"suite": it["suite"], "id": it["id"], "effort": effort, "trial": trial, **gen}
        rec["reasoning_tokens"] = ntok(gen.get("reasoning", ""))
        rec["answer_tokens"] = ntok(gen.get("content", ""))
        return it, rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            it, rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[0] += 1
                if done[0] % 20 == 0 or done[0] == len(jobs):
                    el = time.time() - t_start
                    log(f"  gen {done[0]}/{len(jobs)}  {el/60:.1f} min  "
                        f"eta {el/done[0]*(len(jobs)-done[0])/60:.1f} min")
            _pending.append((it, rec))
    fh.close()

    # ---- phase 2: grade (CPU/subprocess-bound, bounded parallelism)
    log(f"grading {len(_pending)} generations ...")
    results = []
    with ThreadPoolExecutor(max_workers=args.grade_workers) as ex:
        futs = {ex.submit(grade, it, rec, args.max_tests, args.per_test_timeout,
                          args.problem_budget): (it, rec) for it, rec in _pending}
        for i, f in enumerate(as_completed(futs), 1):
            it, rec = futs[f]
            try:
                g = f.result()
            except Exception as e:
                g = {"score": 0.0, "grade_note": f"grader_error: {e}"[:200]}
            results.append({"suite": it["suite"], "id": it["id"], "effort": rec["effort"],
                            "trial": rec["trial"], "score": g["score"],
                            "passed": g.get("passed"), "total": g.get("total"),
                            "grade_note": g.get("grade_note", ""),
                            "reasoning_tokens": rec.get("reasoning_tokens"),
                            "answer_tokens": rec.get("answer_tokens"),
                            "completion_tokens": rec.get("completion_tokens"),
                            "latency_s": rec.get("latency_s"),
                            "finish_reason": rec.get("finish_reason"),
                            "error": rec.get("error")})
            if i % 25 == 0:
                log(f"  graded {i}/{len(_pending)}")

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {outdir/'results.json'}  ({len(results)} rows)")
    log(f"total wall {(time.time()-t_start)/60:.1f} min")


_pending: list = []

if __name__ == "__main__":
    main()
