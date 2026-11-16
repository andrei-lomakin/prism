"""LLM-as-judge for generation metrics (Stage 2).

Fixed prompts + Ollama via LLMGateway. On parse failure → heuristic fallback
so a flaky local LLM does not kill the whole run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from prism.config import Settings
from prism.llm.gateway import LLMGateway
from prism.rag.generate.grounding import check_citations

_REFUSAL_RE = re.compile(
    r"(нет ответа|не\s+найден|no answer|not in the (knowledge|data)|cannot answer|"
    r"don'?t know|база не содержит|в базе нет)",
    re.I,
)


@dataclass
class JudgeScores:
    faithfulness: float
    answer_relevance: float
    refusal_correct: float | None  # only for unanswerable
    citation_ok: bool
    method: str  # llm | heuristic


JUDGE_SYSTEM = (
    "You are a strict RAG evaluator. Reply with ONE JSON object only, no markdown. "
    'Keys: "faithfulness" (0..1), "answer_relevance" (0..1). '
    "faithfulness=1 if every claim is supported by CONTEXT; 0 if hallucinated. "
    "answer_relevance=1 if the answer addresses the QUESTION; lower if off-topic."
)


def _heuristic(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    qtype: str,
    n_sources: int,
) -> JudgeScores:
    cite = check_citations(answer, n_sources)
    if qtype == "unanswerable":
        refused = bool(_REFUSAL_RE.search(answer))
        return JudgeScores(
            faithfulness=1.0 if refused else 0.0,
            answer_relevance=1.0 if refused else 0.2,
            refusal_correct=1.0 if refused else 0.0,
            citation_ok=True,
            method="heuristic",
        )
    # crude lexical overlap faithfulness
    ctx = " ".join(contexts).lower()
    tokens = [t for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{4,}", answer.lower())]
    if not tokens:
        faith = 0.0
    else:
        faith = sum(1 for t in tokens if t in ctx) / len(tokens)
    q_tokens = set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", question.lower()))
    a_tokens = set(tokens)
    rel = len(q_tokens & a_tokens) / max(1, len(q_tokens))
    return JudgeScores(
        faithfulness=max(0.0, min(1.0, faith)),
        answer_relevance=max(0.0, min(1.0, rel)),
        refusal_correct=None,
        citation_ok=cite.ok,
        method="heuristic",
    )


def judge_answer(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    qtype: str,
    settings: Settings,
    use_llm: bool = True,
) -> JudgeScores:
    n_sources = len(contexts)
    cite = check_citations(answer, n_sources)
    if not use_llm:
        return _heuristic(
            question=question,
            answer=answer,
            contexts=contexts,
            qtype=qtype,
            n_sources=n_sources,
        )

    ctx_block = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1)) or "(empty)"
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_block}\n\nANSWER:\n{answer}\n\nReturn JSON."
    try:
        gateway = LLMGateway(settings)
        result = gateway.chat(
            user_message=user,
            system_prompt=JUDGE_SYSTEM,
            temperature=0.0,
        )
        raw = result.text.strip()
        # tolerate ```json fences
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        payload = json.loads(m.group(0) if m else raw)
        faith = float(payload.get("faithfulness", 0))
        arel = float(payload.get("answer_relevance", 0))
        refusal = None
        if qtype == "unanswerable":
            refusal = 1.0 if _REFUSAL_RE.search(answer) else 0.0
        return JudgeScores(
            faithfulness=max(0.0, min(1.0, faith)),
            answer_relevance=max(0.0, min(1.0, arel)),
            refusal_correct=refusal,
            citation_ok=cite.ok,
            method="llm",
        )
    except Exception:
        h = _heuristic(
            question=question,
            answer=answer,
            contexts=contexts,
            qtype=qtype,
            n_sources=n_sources,
        )
        h.citation_ok = cite.ok
        return h
