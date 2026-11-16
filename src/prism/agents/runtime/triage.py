"""Complexity triage without LLM — heuristic plan sketch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Complexity = Literal["simple", "multi", "verify", "research"]


@dataclass
class TriageResult:
    complexity: Complexity
    roles: list[str]
    reason: str
    plan_steps: list[str]


_COMPARE = re.compile(
    r"\b(сравни|разниц|отличи|vs\.?|versus|compare|difference|pros|cons|таблиц)\b",
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
# No trailing \b — Cyrillic stems like «статьям» must match «стать»
_WEB = re.compile(
    r"(веб|web|стать|свеж|интернет|online|fetch|\burl\b|https?://)",
    re.IGNORECASE,
)


def triage(objective: str) -> TriageResult:
    text = (objective or "").strip()
    wants_web = bool(_WEB.search(text))
    comparative = bool(_COMPARE.search(text) or _MULTI.search(text) or len(text) > 220)

    if wants_web and comparative:
        return TriageResult(
            complexity="research",
            roles=["retriever", "web", "analyst", "critic"],
            reason="comparative + web evidence requested",
            plan_steps=[
                "spawn_subagents(retriever,web,analyst,critic)",
                "await_subagents",
                "critic_revision?",
                "synthesize_table",
                "distill",
            ],
        )
    if comparative:
        return TriageResult(
            complexity="multi",
            roles=["retriever", "analyst", "critic"],
            reason="multi-part or comparative query",
            plan_steps=[
                "spawn_subagents(retriever,analyst,critic)",
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
