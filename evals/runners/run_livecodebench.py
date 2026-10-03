"""LiveCodeBench code-generation suite — a contamination-resistant, execution-graded
coding eval that discriminates FRONTIER coder models.

Why this exists: our in-house textbook-algorithm coding suite (graders/code_exec.py)
saturates — an 80B-class coder aces it, so it stops discriminating at the top. LiveCodeBench
pulls fresh competitive-programming problems (LeetCode / AtCoder / Codeforces) with hidden
test batteries. We filter to `contest_date >= 2025-01-01` (after these models' training
cutoffs → contamination-resistant) and `difficulty == hard` (where the frontier separates),
and grade by *running* the model's code against every hidden test — score = fraction passed
(partial credit → scores spread, which is what keeps the suite discriminating).

Dataset: `livecodebench/code_generation_lite` (HF). datasets>=4.0 dropped loading-script
support, so we pull the raw jsonl shards with `hf_hub_download` and parse them directly.
Only `test5.jsonl` + `test6.jsonl` contain any 2025+ problems (test.jsonl..test4.jsonl top
out at 2024-10), so for the default `--min-date 2025-01-01` we scan just those two (~692 MB).

Test-case encoding (matches LiveCodeBench's own decode):
    public_test_cases  — plain JSON string  →  json.loads(x)
    private_test_cases — base64 -> zlib -> pickle -> JSON string:
        json.loads(pickle.loads(zlib.decompress(base64.b64decode(x.encode("utf-8")))))
Each test is {"input", "output", "testtype"} where testtype is:
    "functional" — LeetCode-style: complete the `class Solution` method; `input` is the args
                   (one JSON literal per line), `output` is the expected return value.
    "stdin"      — AtCoder/Codeforces-style: a full program that reads stdin, writes stdout;
                   `input` is fed on stdin, compare stdout (whitespace-normalized) to `output`.

Sandboxing (same hardening as graders/code_exec.py): subprocess, wall-clock timeout per
test, CPU/address-space rlimits, throwaway temp cwd, no network granted. A model whose code
errors/crashes/times out just scores 0 on that test — never crashes the runner. A per-problem
wall budget caps worst-case time when a broken solution TLEs every test.

Usage (grade a model endpoint):
    HF_HOME=/mnt/data/hf-cache-tmp python -m runners.run_livecodebench \
        --model local --gateway-url http://localhost:8000/v1 \
        --limit 30 --output-dir results/lcb

The dataset-loading + grading functions are importable (see validate_livecodebench.py, which
proves the functional AND stdin grading paths with hand-written reference solutions).
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import re
import resource
import subprocess
import sys
import tempfile
import textwrap

from sampling import resolve, to_openai_kwargs
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import click

# Reuse the code-block extractor from our execution grader (same fenced-block logic).
from graders.code_exec import extract_code

HF_REPO = "livecodebench/code_generation_lite"
# The only shards containing any contest_date >= 2025-01-01 (verified: test4 tops out at
# 2024-10-05, test5 starts 2024-09-22 and reaches 2025-01-04, test6 is all 2025).
FILES_2025 = ["test5.jsonl", "test6.jsonl"]
# Full shard list per LiveCodeBench release tag (from the repo's code_generation_lite.py).
VERSION_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}

# Preamble made available to every executed solution (LeetCode starters use `List[int]`,
# competitive solutions lean on collections/heapq/etc). Kept import-only; no I/O.
_PREAMBLE = textwrap.dedent(
    """
    import sys, math, re, bisect, heapq, itertools, functools, collections, string, random, operator
    from typing import List, Optional, Tuple, Dict, Set, Any
    from collections import defaultdict, Counter, deque, OrderedDict
    from functools import lru_cache, reduce, cache
    from itertools import permutations, combinations, product, accumulate, groupby
    from heapq import heappush, heappop, heapify, nlargest, nsmallest
    from math import inf, gcd, sqrt, ceil, floor, comb, factorial
    sys.setrecursionlimit(1_000_000)
    """
).strip()


# --------------------------------------------------------------------------------------
# Dataset loading + decode
# --------------------------------------------------------------------------------------

def decode_test_cases(raw: str) -> list[dict]:
    """Decode a LiveCodeBench test-case blob into [{input, output, testtype}, ...].

    public_test_cases are a plain JSON string; private_test_cases are
    base64 -> zlib -> pickle -> JSON. Try plain JSON first, fall back to the encoded path
    (exactly what LiveCodeBench's own loader does)."""
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def _shard_paths(version_tag: str, min_date: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    allowed = VERSION_FILES.get(version_tag, VERSION_FILES["release_v6"])
    # For the default 2025+ cutoff only the two newest shards are relevant — skip the
    # multi-GB historical shards. Widen only if the caller asks for an earlier cutoff.
    if min_date >= "2025-01-01":
        files = [f for f in FILES_2025 if f in allowed]
    else:
        files = allowed
    paths = []
    for f in files:
        paths.append(Path(hf_hub_download(HF_REPO, f, repo_type="dataset")))
    return paths


def load_problems(version_tag: str, min_date: str, difficulties: list[str],
                  limit: int) -> list[dict]:
    """Load, decode, filter and deterministically select LiveCodeBench problems.

    Filter: contest_date >= min_date AND difficulty in `difficulties`. Deterministic
    selection: sort by (contest_date, question_id), take the first `limit`."""
    problems: list[dict] = []
    for path in _shard_paths(version_tag, min_date):
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                if d["contest_date"][:10] < min_date:
                    continue
                if d["difficulty"] not in difficulties:
                    continue
                pub = decode_test_cases(d["public_test_cases"])
                priv = decode_test_cases(d["private_test_cases"])
                # public first (dedup private that repeat a public case)
                seen = {(t["input"], t["output"]) for t in pub}
                tests = list(pub) + [t for t in priv if (t["input"], t["output"]) not in seen]
                func_name = None
                try:
                    func_name = (json.loads(d.get("metadata") or "{}") or {}).get("func_name")
                except Exception:
                    pass
                problems.append({
                    "question_id": d["question_id"],
                    "title": d["question_title"],
                    "platform": d["platform"],
                    "difficulty": d["difficulty"],
                    "contest_date": d["contest_date"],
                    "question_content": d["question_content"],
                    "starter_code": d.get("starter_code") or "",
                    "func_name": func_name,
                    "tests": tests,
                })
    problems.sort(key=lambda p: (p["contest_date"], p["question_id"]))
    return problems[:limit]


# --------------------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------------------

def build_prompt(problem: dict) -> str:
    starter = problem["starter_code"].strip()
    parts = ["You are an expert competitive programmer. Solve the following problem.",
             "", "### Problem", "", problem["question_content"].strip(), ""]
    if starter:
        parts += [
            "### Starter code (complete this class; keep the given signature)",
            "```python", starter, "```", "",
            "Respond with a single self-contained Python code block that completes the "
            "`Solution` class. Do not read from stdin — implement the method.",
        ]
    else:
        parts += [
            "### Format",
            "Write a complete Python program that reads the input from standard input "
            "(stdin) and writes the answer to standard output (stdout).",
            "Respond with a single self-contained Python code block.",
        ]
    return "\n".join(parts)


def _method_name(problem: dict) -> str | None:
    if problem.get("func_name"):
        return problem["func_name"]
    m = re.search(r"def\s+(\w+)\s*\(\s*self", problem["starter_code"])
    return m.group(1) if m else None


# --------------------------------------------------------------------------------------
# Sandboxed execution / grading
# --------------------------------------------------------------------------------------

def _limit():
    # CPU seconds + ~2GB address space; block core dumps. Best-effort (same as code_exec).
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
        resource.setrlimit(resource.RLIMIT_AS, (2_048 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


_SANDBOX_ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}


def _normalize_stdout(s: str) -> str:
    """Whitespace-normalize program output for comparison: strip trailing space per line,
    drop trailing blank lines. Standard competitive-judge normalization."""
    lines = [ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# In-subprocess harness for functional (LeetCode-style) problems: runs the whole test
# battery in one process with a per-test SIGALRM timeout, so one bad test scores 0 without
# killing the battery (partial credit preserved).
_FUNC_HARNESS = textwrap.dedent(
    '''
    {preamble}
    import json, ast, signal, traceback
    # --- model solution ---
    {solution}
    # --- test battery ---
    class _TO(Exception): pass
    def _alarm(sig, frm): raise _TO()
    signal.signal(signal.SIGALRM, _alarm)

    def _parse(s):
        s = s.strip()
        if s == "": return s
        try: return json.loads(s)
        except Exception: return ast.literal_eval(s)

    _TESTS = json.loads({tests_json!r})
    _METHOD = {method!r}
    _out = []
    for _t in _TESTS:
        signal.alarm({per_test})
        try:
            _args = [_parse(ln) for ln in _t["input"].split("\\n") if ln.strip() != ""]
            _got = getattr(Solution(), _METHOD)(*_args)
            try: _exp = json.loads(_t["output"])
            except Exception: _exp = ast.literal_eval(_t["output"])
            _ok = (_got == _exp)
            if not _ok:
                try:
                    if list(_got) == list(_exp): _ok = True
                except Exception: pass
            _out.append([bool(_ok), ""])
        except _TO:
            _out.append([False, "timeout"])
        except Exception:
            _out.append([False, traceback.format_exc().strip().splitlines()[-1][:150]])
        finally:
            signal.alarm(0)
    print("__GRADE__" + json.dumps(_out))
    '''
).strip()


def grade_functional(code: str, tests: list[dict], method: str,
                     per_test: int, wall_budget: float) -> tuple[int, int, list]:
    """Run a functional solution against its test battery. Returns (passed, total, details)."""
    n = len(tests)
    if not method:
        return 0, n, [{"passed": False, "error": "no method name"}] * n
    script = _FUNC_HARNESS.format(
        preamble=_PREAMBLE, solution=code,
        tests_json=json.dumps(tests), method=method, per_test=per_test,
    )
    outer = min(wall_budget, per_test * n + 10)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "run.py")
        with open(p, "w") as fh:
            fh.write(script)
        try:
            proc = subprocess.run([sys.executable, p], cwd=td, capture_output=True,
                                  text=True, timeout=outer, preexec_fn=_limit, env=_SANDBOX_ENV)
        except subprocess.TimeoutExpired:
            return 0, n, [{"passed": False, "error": "battery timeout"}] * n
    line = next((l for l in proc.stdout.splitlines() if l.startswith("__GRADE__")), None)
    if line is None:
        err = (proc.stderr.strip().splitlines() or ["import/compile error"])[-1][:150]
        return 0, n, [{"passed": False, "error": f"solution error: {err}"}] * n
    results = json.loads(line[len("__GRADE__"):])
    passed = sum(1 for ok, _ in results if ok)
    details = [{"passed": ok, "error": msg} for ok, msg in results]
    return passed, n, details


def grade_stdin(code: str, tests: list[dict], per_test: int,
                wall_budget: float) -> tuple[int, int, list]:
    """Run a stdin program against its test battery (one subprocess per test, feeding
    input on stdin). Returns (passed, total, details). Respects a per-problem wall budget:
    once exceeded, remaining tests are marked failed (guards against TLE-on-every-test)."""
    n = len(tests)
    passed = 0
    details = []
    start = time.time()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "solution.py")
        with open(p, "w") as fh:
            fh.write(_PREAMBLE + "\n" + code)
        for t in tests:
            if time.time() - start > wall_budget:
                details.append({"passed": False, "error": "problem wall budget exceeded"})
                continue
            try:
                proc = subprocess.run([sys.executable, p], cwd=td, input=t["input"],
                                      capture_output=True, text=True, timeout=per_test,
                                      preexec_fn=_limit, env=_SANDBOX_ENV)
            except subprocess.TimeoutExpired:
                details.append({"passed": False, "error": "timeout"})
                continue
            if proc.returncode != 0:
                err = (proc.stderr.strip().splitlines() or ["runtime error"])[-1][:150]
                details.append({"passed": False, "error": err})
                continue
            ok = _normalize_stdout(proc.stdout) == _normalize_stdout(t["output"])
            passed += 1 if ok else 0
            details.append({"passed": ok, "error": "" if ok else "wrong output"})
    return passed, n, details


def grade_problem(problem: dict, model_output: str, max_tests: int,
                  per_test: int, wall_budget: float) -> dict:
    """Extract the model's code and grade it against the problem's test battery.
    score = fraction of tests passed (partial credit)."""
    tests = problem["tests"][:max_tests]
    testtype = tests[0]["testtype"] if tests else "stdin"
    code = extract_code(model_output or "")
    if not code.strip():
        return {"score": 0.0, "passed": 0, "total": len(tests), "testtype": testtype,
                "error": "no code in output", "details": []}
    if testtype == "functional":
        passed, total, details = grade_functional(
            code, tests, _method_name(problem), per_test, wall_budget)
    else:
        passed, total, details = grade_stdin(code, tests, per_test, wall_budget)
    score = passed / total if total else 0.0
    return {"score": score, "passed": passed, "total": total,
            "testtype": testtype, "details": details, "code": code}


# --------------------------------------------------------------------------------------
# Model call
# --------------------------------------------------------------------------------------

def call_model(client, model: str, prompt: str, max_tokens: int,
               force_no_think: bool = False, force_think: bool = False):
    """Return (code_text, completion_tokens). PROD-REPRESENTATIVE: thinking is left to the lane's
    own default (reasoning models think, coders don't) under a realistic `max_tokens` budget that
    matches deployment (32k). A reasoning model is NOT punished for using tokens — it's scored on
    the final extracted code. But the budget is a HARD cap: a model that can't emit a solution
    within it scores 0 (a real deployment failure — it would time users out too, "not endless").
    Token usage is returned separately as an efficiency signal, never folded into correctness.
    `force_no_think=True` forces thinking off (base/non-thinking coding path); `force_think=True`
    forces thinking on (for agentic-thinker coders whose lane defaults thinking off, e.g. Laguna)."""
    # Sampling is overridable: some reasoning models (e.g. Ornith-1.5) FAIL TO TERMINATE at the
    # default 0.2 -- they run to the token cap and emit no code. Their cards specify a much higher
    # temperature (Ornith-1.5: 0.6 general / 1.0 to reproduce benchmarks). Defaults are unchanged,
    # so every existing board number stays valid; set LCB_TEMPERATURE/LCB_TOP_P/LCB_TOP_K to match
    # a model's documented sampling and RECORD IT alongside the score.
    s = resolve("LCB")
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        **to_openai_kwargs(s),
    }
    if force_no_think:
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
    elif force_think:
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": True}
    try:
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        # Code lives in `content` (after any </think>) when the model finished. If content is
        # empty the model burned the whole budget mid-think → fall back to reasoning_content
        # (usually no code fence → scores 0, the correct "didn't solve in budget" outcome).
        if "</think>" in content:
            content = content.split("</think>")[-1]
        if not content.strip():
            rc = getattr(msg, "reasoning_content", None) or ""
            psf = getattr(msg, "provider_specific_fields", {}) or {}
            content = rc or psf.get("reasoning_content", "") or psf.get("reasoning", "")
        usage = getattr(resp, "usage", None)
        ntok = getattr(usage, "completion_tokens", 0) if usage else 0
        return content, ntok
    except Exception as e:
        return f"__MODEL_ERROR__ {e}", 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

@click.command()
@click.option("--model", default="local", help="Gateway model name")
@click.option("--gateway-url", default="http://localhost:8000/v1")
@click.option("--api-key", envvar=["GATEWAY_API_KEY", "LITELLM_API_KEY"], default="not-needed")
@click.option("--version-tag", default="release_v6", help="LiveCodeBench release tag")
@click.option("--min-date", default="2025-01-01", help="Keep problems with contest_date >= this (YYYY-MM-DD)")
@click.option("--difficulty", default="hard", type=click.Choice(["hard", "medium", "easy"]),
              help="Primary difficulty (falls back to add 'medium' if <20 hard problems)")
@click.option("--limit", default=30, type=int, help="Max problems (deterministic selection)")
@click.option("--max-tests", default=20, type=int, help="Max test cases graded per problem")
@click.option("--per-test-timeout", default=6, type=int, help="Wall-clock seconds per test")
@click.option("--problem-budget", default=90.0, type=float, help="Max wall seconds spent grading one problem")
@click.option("--max-tokens", default=32768, type=int,
              help="Max output tokens per solution (prod budget; hard cap — no solution within it = 0)")
@click.option("--no-thinking", is_flag=True,
              help="Force thinking off (measure the base/non-thinking coding path). Default: prod-representative "
                   "(the lane's own thinking default under the --max-tokens budget).")
@click.option("--thinking", is_flag=True,
              help="Force thinking ON (for agentic-thinker coders whose lane defaults thinking off, e.g. Laguna). "
                   "Mutually exclusive with --no-thinking.")
@click.option("--output-dir", type=click.Path(), default=None, help="Directory to write result JSON")
def main(model, gateway_url, api_key, version_tag, min_date, difficulty, limit,
         max_tests, per_test_timeout, problem_budget, max_tokens, no_thinking, thinking, output_dir):
    """Run the LiveCodeBench code-generation suite against a model endpoint."""
    from openai import OpenAI

    # Contamination-resistance + difficulty filter, with the medium fallback.
    difficulties = [difficulty]
    problems = load_problems(version_tag, min_date, difficulties, limit)
    if difficulty == "hard" and len(problems) < 20:
        click.echo(f"Only {len(problems)} hard problems — widening to include 'medium'.")
        difficulties = ["hard", "medium"]
        problems = load_problems(version_tag, min_date, difficulties, limit)

    if not problems:
        click.echo("No problems after filter — check --min-date / --version-tag.")
        sys.exit(1)

    dates = [p["contest_date"][:10] for p in problems]
    plat = {}
    diff = {}
    for p in problems:
        plat[p["platform"]] = plat.get(p["platform"], 0) + 1
        diff[p["difficulty"]] = diff.get(p["difficulty"], 0) + 1
    click.echo("=" * 64)
    click.echo(f"LiveCodeBench  version={version_tag}  difficulty={'+'.join(difficulties)}")
    click.echo(f"Problems: {len(problems)}  date range: {min(dates)} -> {max(dates)}")
    click.echo(f"Platform mix: {plat}   Difficulty mix: {diff}")
    click.echo(f"Model: {model} @ {gateway_url}  (max-tests/problem={max_tests}, per-test-timeout={per_test_timeout}s)")
    click.echo("=" * 64)

    # 600s client timeout accommodates a full 32k-token reasoning trace (~200-300s) without
    # cutting generation short of the code block.
    if no_thinking and thinking:
        click.echo("--no-thinking and --thinking are mutually exclusive.")
        sys.exit(1)
    client = OpenAI(base_url=gateway_url, api_key=api_key, timeout=600.0)
    think_mode = ("thinking OFF (forced)" if no_thinking else
                  "thinking ON (forced)" if thinking else "prod-representative (lane default)")
    click.echo(f"Budget: {max_tokens} tok/solution · {think_mode}")
    click.echo(f"Sampling: {resolve('LCB').summary()}")
    results = []
    for i, problem in enumerate(problems, 1):
        prompt = build_prompt(problem)
        t0 = time.time()
        output, ntok = call_model(client, model, prompt, max_tokens,
                                   force_no_think=no_thinking, force_think=thinking)
        gr = grade_problem(problem, output, max_tests, per_test_timeout, problem_budget)
        dt = time.time() - t0
        results.append({
            "task_id": problem["question_id"],
            "title": problem["title"],
            "platform": problem["platform"],
            "difficulty": problem["difficulty"],
            "tokens": ntok,
            "contest_date": problem["contest_date"],
            "testtype": gr["testtype"],
            "avg_score": gr["score"],
            "passed": gr["passed"],
            "total": gr["total"],
            "error": gr.get("error"),
            "duration_s": round(dt, 1),
        })
        nocode = " NO-CODE" if gr.get("error") == "no code in output" else ""
        click.echo(f"[{i:>2}/{len(problems)}] {problem['platform']:>8} {problem['difficulty']:>6} "
                   f"{problem['title'][:40]:<40} {gr['passed']:>2}/{gr['total']:<2} "
                   f"score={gr['score']:.2f}  {ntok:>5}tok  {dt:.0f}s{nocode}")

    mean = sum(r["avg_score"] for r in results) / len(results)
    solved = sum(1 for r in results if r["avg_score"] == 1.0)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    # "no code in output" at the budget = didn't solve within the cap (a real failure), but track
    # it separately so a harness/format regression can't hide as genuine wrong answers.
    nocode_n = sum(1 for r in results if r.get("error") == "no code in output")
    click.echo("=" * 64)
    click.echo(f"Mean score (avg fraction of tests passed): {mean:.3f}")
    click.echo(f"Fully solved (all tests): {solved}/{len(results)}  "
               f"({solved/len(results):.0%})   [pass@1, contamination-resistant hard]")
    click.echo(f"Avg tokens/solution: {avg_tokens:.0f} (efficiency signal, not scored)   "
               f"no-code-in-budget: {nocode_n}/{len(results)}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = {
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suite_type": "livecodebench",
            "config": {
                "version_tag": version_tag, "min_date": min_date,
                "difficulty": difficulties, "limit": limit,
                "max_tests": max_tests, "per_test_timeout": per_test_timeout,
                "max_tokens": max_tokens, "thinking": ("off" if no_thinking else "prod-default"),
                "sampling": resolve("LCB").as_dict(),
            },
            "date_range": [min(dates), max(dates)],
            "platform_mix": plat,
            "difficulty_mix": diff,
            "problems": results,
            "summary": {
                "count": len(results),
                "mean_score": mean,
                "fully_solved": solved,
                "solve_rate": solved / len(results),
                "avg_tokens": round(avg_tokens),
                "no_code_in_budget": nocode_n,
            },
        }
        f = out / "livecodebench_results.json"
        with open(f, "w") as fh:
            json.dump(data, fh, indent=2)
        click.echo(f"Results written to {f}")


if __name__ == "__main__":
    main()
