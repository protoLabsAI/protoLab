#!/usr/bin/env python3
"""Jev-compatible typed decisions on a stock vLLM lane: state in, calibrated distributions out.

Wire format = TypeSafe's `POST /v1/systemone` (same shapes as OpenJev's schema.py, MIT), so the
official typesafe-sdk can point at this server. Three primitives:
  noul   -> P(yes)
  choice -> distribution over named options + confidence
  score  -> distribution over ordered levels + expected level + confidence

Mechanism (one forward pass per question, no decoding, no reasoning channel):
  - render the chat template with thinking OFF, options labelled A), B)… (score: 0), 1)…)
  - /v1/completions with max_tokens=1, logprobs=20, read the label tokens' probabilities,
    renormalise over the valid labels. `label_mass` = raw mass on valid labels before
    renormalising (how in-schema the model was without a grammar).
  - state goes FIRST so every question about one state shares a prefix: vLLM's prefix cache
    turns N questions into one long prefill plus N short ones.

Labels are single tokens for Qwen3.x (A–T, 0–9 verified), and vLLM caps top logprobs at 20,
so choice supports <= 20 options here. Larger option sets are not implemented yet.

Run:  ~/dev/vllm-025/bin/python s1serve.py --port 8070 --backend http://localhost:8041/v1,http://localhost:8042/v1
"""
from __future__ import annotations

import argparse, asyncio, itertools, json, math, os
from typing import Annotated, Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer

TOKENIZER = "/mnt/models/quantized/ukisai-Swift-Qwen3.8-27B-NVFP4"
LETTERS = "ABCDEFGHIJKLMNOPQRST"
MAX_CHOICE, MAX_SCORE = 20, 10
FLOOR = 1e-6  # probability given to a valid label that fell outside the returned top-20

SYSTEM = ("You are a decision engine. You read the state, then answer the question about it by "
          "choosing exactly one of the labelled options. Reply with the option's label only.")

# ── wire schema (TypeSafe /v1/systemone) ──────────────────────────────────────────────────────
JSONContent = str | dict[str, Any] | list[Any]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(_Strict):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]

    @model_validator(mode="after")
    def _n(self):
        if not 1 <= len(self.criteria) <= MAX_CHOICE:
            raise ValueError(f"choice needs 1..{MAX_CHOICE} options on this backend")
        return self


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(..., min_length=1, max_length=MAX_SCORE)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(_Strict):
    state: JSONContent
    model: str = "smart-s1"
    questions: dict[str, Question] = Field(..., min_length=1)


# ── prompting ────────────────────────────────────────────────────────────────────────────────
def text(x: JSONContent | None) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def compile_question(q) -> tuple[str, list[str], list[str]]:
    """-> (question block, label tokens, outcome names)."""
    if q.type == "noul":
        c = q.criteria or NoulCriteria()
        outcomes = ["yes", "no"]
        descs = [text(c.true), text(c.false)]
        labels = ["A", "B"]
    elif q.type == "choice":
        outcomes = list(q.criteria)
        descs = [text(v) for v in q.criteria.values()]
        labels = list(LETTERS[: len(outcomes)])
    else:
        outcomes = [str(i) for i in range(len(q.criteria))]
        descs = [text(v) for v in q.criteria]
        labels = outcomes
    opts = "\n".join(f"{l}) {o}" + (f": {d}" if d and q.type != "score" else (f" {d}" if d else ""))
                     for l, o, d in zip(labels, outcomes, descs))
    kind = {"noul": "Yes or no.", "choice": "Choose one.", "score": "Choose one level."}[q.type]
    block = f"<question>\n{text(q.instructions)}\n{kind}\n</question>\n\nOptions:\n{opts}\n\nLabel:"
    return block, labels, outcomes


class Engine:
    def __init__(self, backends: list[str], model: str, temps: dict[str, float]):
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self.backends = itertools.cycle(backends)
        self.model = model
        self.temps = temps
        self.http = httpx.AsyncClient(timeout=600)

    def render(self, state: str, block: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<state>\n{state}\n</state>\n\n{block}"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    async def one(self, base: str, state: str, q) -> dict:
        block, labels, outcomes = compile_question(q)
        r = await self.http.post(f"{base}/completions", json={
            "model": self.model, "prompt": self.render(state, block),
            "max_tokens": 1, "temperature": 0.0, "logprobs": 20})
        r.raise_for_status()
        d = r.json()
        top = d["choices"][0]["logprobs"]["top_logprobs"][0]
        raw = [math.exp(top[l]) if l in top else 0.0 for l in labels]
        mass = sum(raw)
        t = self.temps.get(q.type, 1.0)
        logits = [math.log(max(p, FLOOR)) / t for p in raw]
        mx = max(logits)
        ex = [math.exp(x - mx) for x in logits]
        probs = [e / sum(ex) for e in ex]
        best = max(range(len(probs)), key=probs.__getitem__)
        if q.type == "noul":
            ans = {"type": "noul", "noul": probs[0]}
        elif q.type == "choice":
            ans = {"type": "choice", "choice": outcomes[best], "confidence": probs[best],
                   "probabilities": dict(zip(outcomes, probs))}
        else:
            ans = {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
                   "confidence": probs[best], "legend": {o: c for o, c in zip(outcomes, q.criteria)},
                   "probabilities": dict(zip(outcomes, probs))}
        ans["label_mass"] = mass  # extension: not in TypeSafe's schema
        return {"answer": ans, "usage": d.get("usage", {})}

    async def evaluate(self, req: SystemOneRequest) -> dict:
        base = next(self.backends)  # one lane per request, so its questions share the prefix cache
        state = text(req.state)
        names = list(req.questions)
        outs = await asyncio.gather(*(self.one(base, state, req.questions[n]) for n in names))
        return {"model": req.model,
                "answers": {n: o["answer"] for n, o in zip(names, outs)},
                "usage": {"input_tokens": sum(o["usage"].get("prompt_tokens", 0) for o in outs),
                          "output_tokens": sum(o["usage"].get("completion_tokens", 0) for o in outs)}}


def make_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="smart-s1")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        return {"models": [{"name": "smart-s1", "release_date": "2026-10-04",
                            "description": "Swift-Qwen3.8-27B-NVFP4, one forward pass, thinking off"}]}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest):
        try:
            return await engine.evaluate(req)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"backend error: {e}")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--backend", default="http://localhost:8041/v1,http://localhost:8042/v1")
    ap.add_argument("--model", default="smart")
    ap.add_argument("--temps", default="{}", help='per-kind temperature, e.g. {"noul":1.3}')
    a = ap.parse_args()
    eng = Engine(a.backend.split(","), a.model, json.loads(a.temps))
    uvicorn.run(make_app(eng), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
