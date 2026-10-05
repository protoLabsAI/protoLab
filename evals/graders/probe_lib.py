#!/usr/bin/env python3
"""Shared probe helpers. Use these instead of hand-rolling request/parse logic.

Exists because one session produced SEVEN false eval results, all from three causes that
every ad-hoc probe rediscovers: starved/overflowed budgets, reasoning-field drift, and
uncontrolled A/B variables. See feedback_eval_harness_selfsabotage.

The rule these encode: **an empty or zero result is a harness bug until proven otherwise.**
"""
from __future__ import annotations
import json, urllib.request

# Adaptive-thinking models spend the whole budget in the reasoning channel below this and
# return EMPTY content with finish_reason=length, which scores as a failure.
MIN_SANE_BUDGET = 2048


class StarvedResponse(RuntimeError):
    """Empty content because the budget ran out — INVALID, not a score of zero."""


def extract_text(msg: dict) -> str:
    """content -> reasoning -> reasoning_content. vLLM 0.25 renamed reasoning_content to
    `reasoning`; readers that only know the old name see nothing and score a false failure."""
    return ((msg.get("content") or "").strip()
            or (msg.get("reasoning") or "").strip()
            or (msg.get("reasoning_content") or "").strip())


def chat(base_url: str, model: str, messages: list, max_tokens: int = 4096,
         timeout: int = 600, strict: bool = True, **kw) -> dict:
    """POST a chat completion. Returns {text, finish_reason, usage, raw}.

    With strict=True (default) raises StarvedResponse when the model produced nothing
    because it hit the cap — the single most common false-failure in this repo.
    """
    if max_tokens < MIN_SANE_BUDGET:
        raise ValueError(
            f"max_tokens={max_tokens} is below MIN_SANE_BUDGET={MIN_SANE_BUDGET}. "
            "An adaptive-thinking model will return EMPTY content and you will record a "
            "false failure. Raise the budget or pass strict=False deliberately.")
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, **kw}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = d["choices"][0]
    text = extract_text(ch["message"])
    fin = ch.get("finish_reason")
    if strict and not text and fin == "length":
        raise StarvedResponse(
            f"empty content with finish_reason=length at max_tokens={max_tokens} "
            f"(completion_tokens={d.get('usage',{}).get('completion_tokens')}). "
            "This is STARVATION, not model failure — raise the budget and re-run.")
    return {"text": text, "finish_reason": fin, "usage": d.get("usage", {}), "raw": d}


def preflight_context(base_url: str, need_output: int, prompt_headroom: int = 8192) -> int:
    """Verify the served window fits the budgets about to be requested.

    A lane at --max-model-len 32768 asked for max_tokens=32768 makes EVERY request 400;
    the suite then records tokens=0 and reports 0.0, which reads as 'model cannot code'.
    Returns the served max_model_len (0 if unknown)."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            ctx = int(json.load(r)["data"][0].get("max_model_len") or 0)
    except Exception:
        return 0
    need = need_output + prompt_headroom
    if 0 < ctx < need:
        raise RuntimeError(
            f"PREFLIGHT FAIL: served max_model_len={ctx} but this run requests up to "
            f"{need_output} output tokens (need >={need} with prompt headroom). Every "
            "request would 400 and the suite would silently score 0.0.")
    return ctx


def assert_same_config(a: dict, b: dict, keys: list, labels=("A", "B")) -> None:
    """Refuse to compare two runs whose configs differ on `keys`.

    A 'thinking-on lifts LCB 0/15 -> 4/15' finding was wrong because one arm silently ran
    difficulty=[hard,medium] and the other [hard]; the 4 solves were mediums the baseline
    never contained. Paired properly, thinking-on was WORSE."""
    diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
    if diff:
        raise RuntimeError(
            f"REFUSING to compare {labels[0]} vs {labels[1]} — configs differ: " +
            "; ".join(f"{k}: {labels[0]}={v[0]!r} {labels[1]}={v[1]!r}" for k, v in diff.items()))
