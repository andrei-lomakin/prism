"""Complexity triage without LLM — heuristic plan sketch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Complexity = Literal["simple", "multi", "verify"]


@dataclass
class TriageResult:
    complexity: Complexity
    roles: list[str]
    reason: str
    plan_steps: list[str]


_COMPARE = re.compile(
    r"\b(сравни|разниц|отличи|vs\.?|versus|compare|difference|pros|cons)\b",
    re.IGNORECASE,
)
_MULTI = re.compile(
    r"\b(и ещё|а также|во-первых|1\)|2\)|перечисл|список|how does .+ and)\b",
    re.IGNORECASE,
)
_VERIFY = re.compile(
    r"\b(проверь|verify|cite|цитат|источник|уверен)\b",
    re.IGNORECASE,
)


def triage(objective: str) -> TriageResult:
    text = (objective or "").strip()
    if _COMPARE.search(text) or _MULTI.search(text) or len(text) > 220:
        return TriageResult(
            complexity="multi",
            roles=["retriever", "critic"],
            reason="multi-part or comparative query",
            plan_steps=[
                "spawn_subagents(retriever,critic)",
                "await_subagents",
                "synthesize",
            ],
        )
    if _VERIFY.search(text):
        return TriageResult(
            complexity="verify",
            roles=["retriever", "critic"],
            reason="verification requested",
            plan_steps=["search_kb", "draft", "verify_citations", "synthesize"],
        )
    return TriageResult(
        complexity="simple",
        roles=["retriever"],
        reason="single-hop factual",
        plan_steps=["search_kb", "synthesize"],
    )
