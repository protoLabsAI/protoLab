#!/usr/bin/env python3
"""Adversarial coherence probe — catches quant/serving corruption that scores can't.

Two failure classes this hunts:
1. Outright corruption (token soup, `!!!` runs, template leakage) — the class the
   2026-07-03 NVFP4 key-mangling bug produced: healthy server, garbage math.
2. Depth degeneration — quant error compounding at long context: repetition
   loops, drift, mid-word garbage appearing only past ~16-32K tokens, invisible
   to every ≤8K gate suite.

Method: build real-text prompts at each depth rung (concatenated project docs —
NOT random tokens), ask for (a) a needle recall, (b) a continuation. Check
outputs with deterministic degeneration detectors, then optionally an
adversarial LLM judge prompted to FIND corruption (JUDGE_GATEWAY_URL/.env).

Usage:
  python graders/verify_coherence.py --base-url http://localhost:8011/v1 \
      --model ornith-9b-nvfp4 --depths 4096,16384,32768,60000 [--judge]

Exit code 0 = all rungs clean; 1 = degeneration detected (fail the release).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import zlib
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
# Real prose corpus: our own docs — always present, never random tokens
CORPUS_FILES = [
    REPO / "CLAUDE.md",
    REPO / "FOCUS.md",
    REPO / "evals" / "PHASE3_RESULTS.md",
]

NEEDLE = "The maintenance code for the auxiliary condenser is JX-{n}-VELVET."


def build_prompt(depth_tokens: int, needle_n: int) -> tuple[str, str]:
    """Concatenate real docs to ~depth_tokens (3.2 chars/tok — markdown-heavy
    text tokenizes denser than prose; 4.0 overshoots context limits), bury a
    needle at ~60% depth, ask recall + continuation."""
    text = ""
    target_chars = int(depth_tokens * 3.2)
    while len(text) < target_chars:
        for f in CORPUS_FILES:
            text += f.read_text(errors="ignore") + "\n\n"
            if len(text) >= target_chars:
                break
    text = text[:target_chars]
    needle = NEEDLE.format(n=needle_n)
    pos = int(len(text) * 0.6)
    text = text[:pos] + f"\n\n{needle}\n\n" + text[pos:]
    question = (
        "\n\n---\nAnswer two things:\n"
        "1. What is the maintenance code for the auxiliary condenser mentioned above?\n"
        "2. In 2-3 sentences, what kind of document is this and what is it about?"
    )
    return text + question, needle


# ── deterministic degeneration detectors ───────────────────────────────────

def detectors(s: str) -> dict:
    out = {}
    if not s or len(s.strip()) < 10:
        return {"empty": True}
    # 1. char-run spam (!!!! or any single-char run)
    longest_run = max(len(m.group(0)) for m in re.finditer(r"(.)\1*", s))
    out["max_char_run"] = longest_run
    # 2. n-gram loop: most frequent 8-gram (word) share
    words = s.split()
    if len(words) >= 24:
        grams = [" ".join(words[i : i + 8]) for i in range(len(words) - 8)]
        from collections import Counter

        top = Counter(grams).most_common(1)[0][1]
        out["ngram8_max_repeat"] = top
    # 3. compression ratio — degenerate text compresses absurdly well
    comp = len(zlib.compress(s.encode())) / max(1, len(s.encode()))
    out["compression_ratio"] = round(comp, 3)
    # 4. non-printable / replacement-char density
    weird = sum(1 for c in s if ord(c) == 0xFFFD or (ord(c) < 32 and c not in "\n\t\r"))
    out["weird_char_frac"] = round(weird / len(s), 4)
    # 5. template/think leakage
    out["template_leak"] = bool(
        re.search(r"<\|im_(start|end)\|>|<\|channel\|?>|(?<!\S)<think>", s)
    )
    return out


def is_degenerate(d: dict) -> list[str]:
    flags = []
    if d.get("empty"):
        flags.append("empty output")
    if d.get("max_char_run", 0) > 12:
        flags.append(f"char run x{d['max_char_run']}")
    if d.get("ngram8_max_repeat", 0) > 4:
        flags.append(f"8-gram loop x{d['ngram8_max_repeat']}")
    if d.get("compression_ratio", 1) < 0.18:
        flags.append(f"compression {d['compression_ratio']}")
    if d.get("weird_char_frac", 0) > 0.005:
        flags.append(f"weird chars {d['weird_char_frac']}")
    if d.get("template_leak"):
        flags.append("template/think leakage")
    return flags


ADVERSARIAL_JUDGE_PROMPT = """You are auditing output from a quantized LLM for CORRUPTION. \
Be adversarial: your job is to find evidence of degeneration, not to be charitable. \
Flag ANY of: repeated phrases/loops, mid-word garbage, wrong-language switches, \
token soup, truncated words, chat-template artifacts, incoherent grammar that a \
healthy model would not produce. Subtle drift counts. Respond with JSON only: \
{"corrupt": true/false, "evidence": "<quote the worst passage or 'none'>"}"""


def judge(client_judge, judge_model: str, text: str) -> dict:
    r = client_judge.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": ADVERSARIAL_JUDGE_PROMPT},
            {"role": "user", "content": text[:6000]},
        ],
        max_tokens=300,
        temperature=0,
    )
    raw = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {"corrupt": None, "evidence": raw[:200]}
    except Exception:
        return {"corrupt": None, "evidence": raw[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="4096,16384,32768,60000")
    # 2500 default: thinking models can spend >600 tokens reasoning before emitting
    # content — a lower cap reads as "empty output" and false-fails the gate
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--judge", action="store_true", help="also run adversarial LLM judge")
    # Context ceiling used to size the generation budget. Was hardcoded to 64512, which
    # silently starved every depth past ~62K to a 256-token budget -- on an adaptive-thinking
    # model that returns EMPTY content and the gate reported "FAIL empty output" on a model
    # that was actually needle-exact at 200K. Set this to the served --max-model-len.
    ap.add_argument("--ctx", type=int, default=262144)
    # Never generate below this: it is the floor at which a thinking model can still emit
    # content rather than spending the whole budget in the reasoning channel.
    ap.add_argument("--min-budget", type=int, default=2048)
    # A rare failure (protoLab#36: ~4% at 60-80K) is invisible to one probe per rung.
    ap.add_argument("--trials", type=int, default=1, help="probes per depth rung")
    ap.add_argument("--reasoning-effort", default=None, help="e.g. low (the #36 failing shape)")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    jclient = jmodel = None
    if args.judge:
        jurl = os.environ.get("JUDGE_GATEWAY_URL", "http://localhost:8000/v1")
        jmodel = os.environ.get("JUDGE_MODEL", "local")
        jclient = OpenAI(base_url=jurl, api_key=os.environ.get("GATEWAY_API_KEY", "not-needed"))

    failed = False
    rungs = [int(x) for x in args.depths.split(",")]
    early_eos = {d: 0 for d in rungs}
    for i, d in enumerate(rungs * args.trials):
        prompt, needle = build_prompt(d, needle_n=1000 + i)
        # Auto-cap generation so depth + budget never overflows the context window
        # (learned on A1: 60K rung + 6K budget = 400 error) -- but never below
        # --min-budget, because a starved budget is indistinguishable from corruption.
        headroom = args.ctx - d
        if headroom < args.min_budget:
            print(f"depth {d:>6}: SKIP — needs {args.min_budget} tokens of headroom, "
                  f"ctx {args.ctx} leaves {headroom}. Raise --ctx or drop this depth.")
            continue
        budget = max(args.min_budget, min(args.max_tokens, headroom))
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=budget,
                **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}),
            )
        except Exception as e:
            # Probe/infra error ≠ model degeneration — report distinctly, still fail the gate
            print(f"depth {d:>6}: PROBE-ERROR {str(e)[:120]}")
            failed = True
            continue
        msg = r.choices[0].message
        # vLLM 0.25 renamed reasoning_content -> reasoning; read both or the fallback misses.
        text = ((msg.content or "")
                + (getattr(msg, "reasoning", None) or "")
                + (getattr(msg, "reasoning_content", None) or ""))
        # STARVED != FAILED. Empty output because the cap ran out is a harness problem, and
        # reporting it as degeneration is how this gate false-failed a model that was
        # needle-exact at 200K. Report INVALID and do not let it score.
        if not text.strip() and getattr(r.choices[0], "finish_reason", None) == "length":
            print(f"depth {d:>6}: INVALID — empty output at budget {budget} "
                  f"(finish_reason=length). Starved, not degenerate: raise --min-budget/--max-tokens.")
            failed = True
            continue
        # EARLY EOS (protoLab#36): reasoning present, no answer, and the model STOPPED on
        # its own (finish_reason=stop, nowhere near the cap). Not starvation, not a parser
        # misroute -- the reply ends mid-reasoning, often mid-token. Report it by name.
        reasoning = (getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or "")
        if (len((msg.content or "").strip()) < 10 and reasoning.strip()
                and getattr(r.choices[0], "finish_reason", None) == "stop"):
            early_eos[d] += 1
            failed = True
            ctoks = getattr(getattr(r, "usage", None), "completion_tokens", "?")
            print(f"depth {d:>6}: FAIL early EOS mid-reasoning — {ctoks} completion tokens, "
                  f"budget {budget}, tail {reasoning[-60:]!r}")
            continue
        det = detectors(msg.content or "")
        flags = is_degenerate(det)
        code = re.search(r"JX-\d+-VELVET", text)
        recall = "✓" if (code and code.group(0) == needle.split()[-1].rstrip(".")) else (
            "✓" if needle.split()[-1].rstrip(".") in text else "✗"
        )
        verdict = "FAIL " + "; ".join(flags) if flags else "clean"
        jnote = ""
        if jclient and not flags:
            jv = judge(jclient, jmodel, msg.content or "")
            if jv.get("corrupt"):
                verdict, failed = f"JUDGE-FLAG {jv['evidence'][:80]}", True
            jnote = " (judged)"
        if flags:
            failed = True
        print(f"depth {d:>6}: needle={recall} detectors={verdict}{jnote}")
        print(f"    sample: {(msg.content or '')[:140]!r}")

    if args.trials > 1 or any(early_eos.values()):
        print("early-EOS per rung: " + ", ".join(
            f"{d}={early_eos[d]}/{args.trials}" for d in rungs))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
