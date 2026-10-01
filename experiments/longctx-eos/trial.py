#!/usr/bin/env python3
"""protoLab#36 repro: early EOS mid-reasoning on 60-80K-token prompts.

Matches the failing shape from Vera's review calls: real code text, non-streaming,
strict json_schema, reasoning_effort=low. Each trial is a distinct, cold prompt
(a nonce leads, so no prefix-cache hit). Seeds are fixed, so every variant gets the
SAME prompts and the arms are paired.

Usage:
  python trial.py --base-url http://localhost:8060/v1 --label k3 --n 120 --conc 8
Writes results/<label>.jsonl, one row per trial, and resumes if the file exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

MODEL_DIR = "/mnt/models/quantized/ukisai-Swift-Qwen3.8-27B-NVFP4"
CORPUS = sorted(glob.glob(
    "/home/ava/dev/vllm-025/lib/python3.12/site-packages/vllm/**/*.py", recursive=True))
OUT = Path(__file__).parent / "results"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "findings"],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "findings": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["file", "severity", "summary"],
            "properties": {
                "file": {"type": "string"},
                "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                "summary": {"type": "string"},
            },
        }},
    },
}

SYSTEM = ("You are a senior code reviewer. Review the files in the user's message for "
          "correctness bugs. Report at most 8 findings. Answer only with the JSON object.")


def build_prompt(tok, seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    target = rng.randint(65_000, 80_000)
    files = CORPUS[:]
    rng.shuffle(files)
    parts, ntok = [f"Review request {seed:06d}-{rng.getrandbits(48):012x}\n\n"], 0
    for f in files:
        body = Path(f).read_text(errors="ignore")
        if not body.strip():
            continue
        chunk = f"=== FILE: {f.split('site-packages/')[-1]} ===\n{body}\n\n"
        n = len(tok(chunk).input_ids)
        if ntok + n > target:
            if target - ntok < 2_000:
                break
            continue  # too big for what's left; a smaller file may still fit
        parts.append(chunk)
        ntok += n
    parts.append("Review these files and return the JSON verdict.")
    return "".join(parts), target


def run_one(client, model, effort, seed, prompt, schema=True):
    t0 = time.time()
    row = {"seed": seed}
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            reasoning_effort=effort,
            **({"response_format": {"type": "json_schema", "json_schema": {
                "name": "review", "strict": True, "schema": SCHEMA}}} if schema else {}),
            max_tokens=32768,
            timeout=3600,
        )
    except Exception as e:  # infra error, not a model outcome
        row.update(error=str(e)[:300], secs=round(time.time() - t0, 1))
        return row
    c = r.choices[0]
    msg = c.message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
    try:
        json.loads(content)
        valid = True
    except Exception:
        valid = False
    row.update(
        prompt_tokens=r.usage.prompt_tokens,
        completion_tokens=r.usage.completion_tokens,
        finish_reason=c.finish_reason,
        stop_reason=getattr(c, "stop_reason", None),
        content_chars=len(content),
        reasoning_chars=len(reasoning),
        json_valid=valid,
        early_eos=(len(content.strip()) < 10 and bool(reasoning.strip())
                   and c.finish_reason == "stop"),
        content=content,
        reasoning_tail=reasoning[-120:],
        secs=round(time.time() - t0, 1),
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="smart")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--no-schema", action="store_true", help="drop response_format (isolates guided decoding)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    out = OUT / f"{args.label}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["seed"] for l in out.open() if l.strip()
                and "error" not in json.loads(l)}
    seeds = [s for s in range(args.seed0, args.seed0 + args.n) if s not in done]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    client = OpenAI(base_url=args.base_url, api_key="x")
    print(f"{args.label}: {len(seeds)} trials to run ({len(done)} already done)", flush=True)

    eos = tot = 0
    with ThreadPoolExecutor(args.conc) as ex, out.open("a") as fh:
        futs = {ex.submit(run_one, client, args.model, args.effort, s,
                          build_prompt(tok, s)[0], not args.no_schema): s for s in seeds}
        for f in as_completed(futs):
            row = f.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if "error" in row:
                print(f"  seed {row['seed']}: ERROR {row['error'][:100]}", flush=True)
                continue
            tot += 1
            eos += row["early_eos"]
            flag = " EARLY-EOS" if row["early_eos"] else ""
            print(f"  seed {row['seed']:>4}: {row['prompt_tokens']}p {row['completion_tokens']}c "
                  f"{row['finish_reason']} json={row['json_valid']}{flag}  [{eos}/{tot}]", flush=True)


if __name__ == "__main__":
    main()
