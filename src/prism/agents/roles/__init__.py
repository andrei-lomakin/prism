"""Agent roles: Orchestrator, Retriever, Critic (+ stubs web/analyst)."""

from __future__ import annotations

from typing import Any

from prism.agents.policy.gate import ActionGate
from prism.agents.runtime.limits import LimitTracker
from prism.agents.tools import ToolRegistry

ROLES = ("orchestrator", "retriever", "web", "analyst", "critic")


def run_retriever(
    *,
    gate: ActionGate,
    registry: ToolRegistry,
    limits: LimitTracker,
    objective: str,
    run: Any,
) -> dict[str, Any]:
    """Single search_kb via gate; returns hits."""
    tool = registry.get("search_kb")
    assert tool is not None
    args = {"query": objective}
    if limits.is_duplicate_read("search_kb", args, side_effect=tool.side_effect):
        return {"hits": [], "dedup": True, "role": "retriever"}
    limits.note_step()
    limits.note_search()
    result = gate.execute(run=run, tool=tool, args=args)
    if not result.executed:
        return {"hits": [], "error": result.error, "role": "retriever"}
    out = dict(result.output or {})
    out["role"] = "retriever"
    return out


def run_critic(
    *,
    gate: ActionGate,
    registry: ToolRegistry,
    limits: LimitTracker,
    objective: str,
    answer: str,
    sources: list[str],
    run: Any,
) -> dict[str, Any]:
    """verify_citations on a draft answer."""
    tool = registry.get("verify_citations")
    assert tool is not None
    args = {"answer": answer, "sources": sources}
    if limits.is_duplicate_read("verify_citations", args, side_effect=tool.side_effect):
        return {"dedup": True, "role": "critic", "grounding_ok": True}
    limits.note_step()
    result = gate.execute(run=run, tool=tool, args=args)
    if not result.executed:
        return {"error": result.error, "role": "critic", "grounding_ok": False}
    out = dict(result.output or {})
    out["role"] = "critic"
    out["objective"] = objective
    return out
