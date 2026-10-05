"""The decision prompt, shared by the server, the trainer, the teacher labeller and the evals.

Anything that changes the rendered text changes what a fine-tuned model was trained on, so all
of them import from here and nothing re-implements it.
"""
from __future__ import annotations

import json
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOPQRST"
MAX_CHOICE, MAX_SCORE = 20, 10

SYSTEM = ("You are a decision engine. You read the state, then answer the question about it by "
          "choosing exactly one of the labelled options. Reply with the option's label only.")


def text(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def compile_question(q: dict, order: list[int] | None = None) -> tuple[str, list[str], list[str]]:
    """TypeSafe question dict -> (question block, label tokens, outcome names).

    `order` permutes choice options (training-time shuffling). noul and score keep their fixed
    order: yes/no, and levels 0..k-1 (the level IS the label).
    """
    t = q["type"]
    if t == "noul":
        c = q.get("criteria") or {}
        outcomes, descs, labels = ["yes", "no"], [text(c.get("true")), text(c.get("false"))], ["A", "B"]
    elif t == "choice":
        items = list(q["criteria"].items())
        if order is not None:
            items = [items[i] for i in order]
        outcomes = [k for k, _ in items]
        descs = [text(v) for _, v in items]
        labels = list(LETTERS[: len(outcomes)])
    elif t == "score":
        outcomes = [str(i) for i in range(len(q["criteria"]))]
        descs = [text(v) for v in q["criteria"]]
        labels = outcomes
    else:
        raise ValueError(f"unknown question type {t!r}")
    opts = "\n".join(f"{l}) {o}" + (f": {d}" if d and t != "score" else (f" {d}" if d else ""))
                     for l, o, d in zip(labels, outcomes, descs))
    kind = {"noul": "Yes or no.", "choice": "Choose one.", "score": "Choose one level."}[t]
    block = f"<question>\n{text(q.get('instructions'))}\n{kind}\n</question>\n\nOptions:\n{opts}\n\nLabel:"
    return block, labels, outcomes


def messages(state: Any, block: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<state>\n{text(state)}\n</state>\n\n{block}"}]


def render(tok, state: Any, block: str) -> str:
    return tok.apply_chat_template(messages(state, block), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
