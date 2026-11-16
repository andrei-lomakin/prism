"""Grounded generation prompts with numbered sources [n]."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "Ты research-ассистент. Отвечай только по предоставленному контексту. "
    "Каждое фактическое утверждение снабжай ссылкой вида [n] на номер источника. "
    "Если в контексте нет ответа — прямо скажи, что в базе нет ответа. "
    "Отвечай на языке вопроса."
)


def build_rag_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        blocks = "(контекст пуст — релевантные фрагменты не найдены)"
    else:
        parts: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            src = ctx.get("document_name") or "документ"
            score = ctx.get("score")
            score_s = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
            section = ctx.get("section_path")
            section_s = f", section={section}" if section else ""
            text = (ctx.get("text") or "").strip()
            parts.append(f"[{i}] Источник: {src}{score_s}{section_s}\n{text}")
        blocks = "\n\n".join(parts)

    return (
        "Контекст из базы знаний:\n"
        f"{blocks}\n\n"
        "Вопрос:\n"
        f"{question}\n\n"
        "Сформулируй ответ с цитатами [n]. Если ответа нет в контексте — откажись."
    )
