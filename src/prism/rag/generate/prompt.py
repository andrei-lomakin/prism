"""Grounded generation prompts with numbered sources [n]."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "Ты research-ассистент. Отвечай только по предоставленному контексту. "
    "Каждое фактическое утверждение снабжай ссылкой вида [n] на номер источника. "
    "Если в контексте нет ответа — прямо скажи, что в базе нет ответа. "
    "Отвечай на языке вопроса."
)


def build_rag_prompt(
    question: str,
    contexts: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
) -> str:
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

    dialog = ""
    turns = [
        t for t in (history or []) if isinstance(t, dict) and (t.get("content") or "").strip()
    ][-6:]
    if turns:
        lines = []
        for t in turns:
            role = "Пользователь" if t.get("role") == "user" else "Ассистент"
            lines.append(f"{role}: {str(t.get('content') or '').strip()[:1200]}")
        dialog = "Предыдущий диалог:\n" + "\n".join(lines) + "\n\n"

    return (
        "Контекст из базы знаний:\n"
        f"{blocks}\n\n"
        f"{dialog}"
        "Текущий вопрос:\n"
        f"{question}\n\n"
        "Сформулируй ответ с цитатами [n]. Учитывай диалог, но опирайся на контекст. "
        "Если ответа нет в контексте — откажись."
    )
