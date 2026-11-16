"""Agent roles: Orchestrator, Retriever, Web, Analyst, Critic."""

from __future__ import annotations

from typing import Any

from prism.agents.policy.gate import ActionGate
from prism.agents.runtime.limits import LimitTracker
from prism.agents.tools import ToolRegistry

ROLES = ("orchestrator", "retriever", "web", "analyst", "critic")


def _call(
    *,
    gate: ActionGate,
    registry: ToolRegistry,
    limits: LimitTracker,
    run: Any,
    tool_code: str,
    args: dict[str, Any],
    count_search: bool = False,
) -> dict[str, Any]:
    tool = registry.get(tool_code)
    if tool is None:
        return {"error": f"missing_tool:{tool_code}"}
    if limits.is_duplicate_read(tool_code, args, side_effect=tool.side_effect):
        return {"dedup": True, "tool": tool_code}
    limits.note_step()
    if count_search:
        limits.note_search()
    result = gate.execute(run=run, tool=tool, args=args)
    if not result.executed:
        return {
            "error": result.error,
            "tool": tool_code,
            "approval_id": result.approval_id,
            "mode": result.decision.mode,
        }
    out = dict(result.output) if isinstance(result.output, dict) else {"value": result.output}
    out["tool"] = tool_code
    return out


def run_retriever(
    *,
    gate: ActionGate,
    registry: ToolRegistry,
    limits: LimitTracker,
    objective: str,
    run: Any,
) -> dict[str, Any]:
    out = _call(
        gate=gate,
        registry=registry,
        limits=limits,
        run=run,
        tool_code="search_kb",
        args={"query": objective},
        count_search=True,
    )
    out["role"] = "retriever"
    out.setdefault("hits", [])
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
    out = _call(
        gate=gate,
        registry=registry,
        limits=limits,
        run=run,
        tool_code="verify_citations",
        args={"answer": answer, "sources": sources},
    )
    # Optional soft judge
    if out.get("error") is None and registry.get("judge") is not None:
        judged = _call(
            gate=gate,
            registry=registry,
            limits=limits,
            run=run,
            tool_code="judge",
            args={"answer": answer, "question": objective, "sources": sources},
        )
        if judged.get("error") is None:
            out["judge"] = judged
            if judged.get("ok") is False:
                out["grounding_ok"] = False
                out["needs_revision"] = True
    out["role"] = "critic"
    out["objective"] = objective
    if out.get("grounding_ok") is False:
        out["needs_revision"] = True
    return out


def run_web(
    *,
    gate: ActionGate,
    registry: ToolRegistry,
    limits: LimitTracker,
    objective: str,
    run: Any,
) -> dict[str, Any]:
    out = _call(
        gate=gate,
        registry=registry,
        limits=limits,
        run=run,
        tool_code="web_search",
        args={"query": objective},
        count_search=True,
    )
    out["role"] = "web"
    # Optionally fetch first URL if search returned links and gate allows
    results = list(out.get("results") or [])
    pages: list[dict[str, Any]] = []
    for item in results[:2]:
        url = item.get("url")
        if not url:
            continue
        page = _call(
            gate=gate,
            registry=registry,
            limits=limits,
            run=run,
            tool_code="fetch_page",
            args={"url": url},
        )
        pages.append(page)
        if page.get("error") == "approval required":
            out["approval_id"] = page.get("approval_id")
            out["error"] = "approval required"
            break
    if pages:
        out["pages"] = pages
    return out


def run_analyst(
    *,
    gate: ActionGate,
    registry: ToolRegistry,
    limits: LimitTracker,
    objective: str,
    run: Any,
    kb_hits: list[dict[str, Any]] | None = None,
    web_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out = _call(
        gate=gate,
        registry=registry,
        limits=limits,
        run=run,
        tool_code="build_comparison_table",
        args={
            "objective": objective,
            "kb_hits": kb_hits or [],
            "web_results": web_results or [],
        },
    )
    out["role"] = "analyst"
    return out
