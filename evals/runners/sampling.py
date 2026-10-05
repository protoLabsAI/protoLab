"""One place where every runner resolves its sampling — and records it.

Why this exists (2026-08-23): sampling was inconsistent and invisible across the
suite. `run_function_call` and `run_rag` sent `presence_penalty 1.5`;
`run_livecodebench` and `run_ctibench` sent none. A model therefore got different
sampling depending on which suite it was in, and **nothing was written to
`scorecard.json`** — the only record was a line in `run.log`, which no aggregator
reads. Reconstructing what a board number was measured at meant grepping logs.

That matters more than it sounds for models whose card specifies sampling.
Ornith-1.5 documents temp 0.6 (precise coding) / 1.0 + presence_penalty 1.5
(general); our LiveCodeBench default of 0.2 with no penalty is neither, and
measurably doubles its token burn and induces repetition loops (see
`experiments/quantize/looping/`).

**Defaults here reproduce each runner's prior behaviour exactly**, so existing
board numbers stay valid and comparable. The change is that sampling is now
resolvable from one place, overridable per-suite by env, and *recorded*.

    from sampling import resolve, to_openai_kwargs
    s = resolve("LCB")                       # honours LCB_TEMPERATURE etc.
    kwargs.update(to_openai_kwargs(s))
    ...
    data["config"]["sampling"] = s.as_dict() # lands in scorecard.json
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

# Per-suite defaults = exactly what each runner sent before this module existed.
# Do NOT "harmonise" these silently: changing one invalidates that suite's board
# history. Change deliberately, re-baseline, and say so in the commit.
_DEFAULTS = {
    "LCB": dict(temperature=0.2, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0),
    "CTI": dict(temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0),
    "FC": dict(temperature=0.0, top_p=0.8, top_k=20, min_p=0.0,
               presence_penalty=1.5, repetition_penalty=1.0),
    "RAG": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                presence_penalty=1.5, repetition_penalty=1.0),
}


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    source: str          # which suite's defaults, and whether env overrode them

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"temp={self.temperature} top_p={self.top_p} top_k={self.top_k} "
                f"min_p={self.min_p} presence_penalty={self.presence_penalty} "
                f"repetition_penalty={self.repetition_penalty} [{self.source}]")


def resolve(suite: str) -> Sampling:
    """Resolve sampling for a suite, honouring `<SUITE>_<PARAM>` env overrides.

    e.g. LCB_TEMPERATURE, LCB_PRESENCE_PENALTY, FC_TOP_P.
    """
    key = suite.upper()
    if key not in _DEFAULTS:
        raise KeyError(f"unknown suite {suite!r}; known: {sorted(_DEFAULTS)}")
    base = dict(_DEFAULTS[key])
    overridden = []
    casts = {"top_k": int}
    for name in base:
        env = f"{key}_{name.upper()}"
        if env in os.environ:
            base[name] = casts.get(name, float)(os.environ[env])
            overridden.append(name)
    source = f"{key} defaults" if not overridden else \
             f"{key} defaults + env override: {','.join(sorted(overridden))}"
    return Sampling(source=source, **base)


def to_openai_kwargs(s: Sampling) -> dict:
    """Split a Sampling into OpenAI-standard fields and vLLM `extra_body` fields."""
    return {
        "temperature": s.temperature,
        "top_p": s.top_p,
        "presence_penalty": s.presence_penalty,
        "extra_body": {
            "top_k": s.top_k,
            "min_p": s.min_p,
            "repetition_penalty": s.repetition_penalty,
        },
    }
