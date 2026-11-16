"""Query rewriting strategies R0–R5."""

from __future__ import annotations

from typing import Any

from prism.config import Settings, get_settings


def rewrite(
    query: str,
    strategy: str = "R0",
    history: list[dict] | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return semantic_query, lexical_query, and extra queries for multi-query."""
    strategy = strategy or "R0"
    history = history or []
    cfg = settings or get_settings()

    if strategy == "R0":
        return {
            "strategy": "R0",
            "semantic_query": query,
            "lexical_query": query,
            "queries": [query],
        }
    if strategy == "R1":
        return _r1_condense(query, history, cfg)
    if strategy == "R2":
        return _r2_multi(query, cfg)
    if strategy == "R3":
        return _r3_hyde(query, cfg)
    if strategy == "R4":
        # Semantic with dialog context; lexical = last user turn only
        semantic = _r1_condense(query, history, cfg)["semantic_query"] if history else query
        return {
            "strategy": "R4",
            "semantic_query": semantic,
            "lexical_query": query,
            "queries": [semantic],
        }
    if strategy == "R5":
        return _r5_planner(query, cfg)
    raise ValueError(f"Unknown rewrite strategy: {strategy}")


def _r1_condense(query: str, history: list[dict], settings: Settings) -> dict[str, Any]:
    if not history:
        return {
            "strategy": "R1",
            "semantic_query": query,
            "lexical_query": query,
            "queries": [query],
        }
    # Cheap deterministic condense without LLM when history is short
    prior = " ".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-4:])
    try:
        from prism.llm.gateway import LLMGateway

        gw = LLMGateway(settings)
        prompt = (
            "Given the dialog, rewrite the follow-up into ONE standalone search query.\n"
            f"Dialog:\n{prior}\n\nFollow-up: {query}\n\nStandalone query:"
        )
        out = gw.chat(
            user_message=prompt,
            system_prompt="Return only the standalone query text.",
            temperature=0.0,
        ).text.strip()
        standalone = out or query
    except Exception:
        standalone = f"{prior}\n{query}"[-500:]
    return {
        "strategy": "R1",
        "semantic_query": standalone,
        "lexical_query": query,
        "queries": [standalone],
    }


def _r2_multi(query: str, settings: Settings) -> dict[str, Any]:
    variants = [query]
    try:
        from prism.llm.gateway import LLMGateway

        gw = LLMGateway(settings)
        prompt = (
            "Produce exactly 3 diverse search paraphrases of the question, "
            "one per line, no numbering.\n"
            f"Question: {query}"
        )
        text = gw.chat(
            user_message=prompt,
            system_prompt="Output 3 lines only.",
            temperature=0.4,
        ).text
        lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
        variants = [query] + [ln for ln in lines if ln and ln.lower() != query.lower()][:3]
    except Exception:
        # Deterministic paraphrases without LLM
        variants = [
            query,
            f"explain {query}",
            f"definition of {query}",
            f"details about {query}",
        ]
    # unique preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return {
        "strategy": "R2",
        "semantic_query": uniq[0],
        "lexical_query": query,
        "queries": uniq[:4],
    }


def _r3_hyde(query: str, settings: Settings) -> dict[str, Any]:
    hypo = query
    try:
        from prism.llm.gateway import LLMGateway

        gw = LLMGateway(settings)
        hypo = (
            gw.chat(
                user_message=(
                    "Write a short hypothetical answer passage (3-5 sentences) "
                    f"that would contain evidence for: {query}"
                ),
                system_prompt="Write only the passage.",
                temperature=0.3,
            ).text.strip()
            or query
        )
    except Exception:
        hypo = f"The answer discusses {query} with definitions, steps, and examples."
    return {
        "strategy": "R3",
        "semantic_query": hypo,
        "lexical_query": query,
        "queries": [hypo],
        "hyde_passage": hypo,
    }


def _r5_planner(query: str, settings: Settings) -> dict[str, Any]:
    """Rule-based intent → source budget (LLM optional polish)."""
    q = query.lower()
    intent = "factoid"
    if any(w in q for w in ("compare", "vs", "versus", "сравни", "отличи")):
        intent = "comparison"
    elif any(w in q for w in ("how many", "сколько", "число", "α", "alpha", "k=")):
        intent = "numeric"
    elif any(w in q for w in ("why", "почему", "зачем")):
        intent = "explanation"

    budgets = {
        "factoid": {"top_k": 5, "prefer": "dense"},
        "comparison": {"top_k": 8, "prefer": "hybrid"},
        "numeric": {"top_k": 5, "prefer": "hybrid"},
        "explanation": {"top_k": 8, "prefer": "dense"},
    }
    plan = budgets[intent]
    return {
        "strategy": "R5",
        "semantic_query": query,
        "lexical_query": query,
        "queries": [query],
        "intent": intent,
        "plan": plan,
    }
