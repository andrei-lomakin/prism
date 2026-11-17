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
            "outcome": result.outcome,
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
    if out.get("error") == "approval required":
        out["phase"] = "web_search"
        out["results"] = []
        out["fetch_queue"] = []
        return out
    status = str(out.get("status") or ("ok" if out.get("results") else "empty"))
    out["status"] = status
    if out.get("dedup") or status in ("error", "empty") or out.get("ok") is False:
        out["results"] = []
        out["web_unavailable"] = True
        out["phase"] = "web_done"
        out["fetch_queue"] = []
        return out
    results = list(out.get("results") or [])
    queue: list[str] = []
    for item in results:
        url = item.get("url")
        if url and url not in queue:
            queue.append(str(url))
        if len(queue) == 2:
            break
    pages: list[dict[str, Any]] = []
    for index, url in enumerate(queue):
        page = _call(
            gate=gate,
            registry=registry,
            limits=limits,
            run=run,
            tool_code="fetch_page",
            args={"url": url},
        )
        if page.get("error") == "approval required":
            out["phase"] = "fetch_page"
            out["fetch_queue"] = queue[index:]
            out["pages"] = pages
            out["results"] = results
            out["error"] = "approval required"
            out["approval_id"] = page.get("approval_id")
            out["pending_tool"] = "fetch_page"
            out["pending_args"] = {"url": url}
            return out
        pages.append(page)
    out["pages"] = pages
    out["results"] = results
    out["phase"] = "web_done"
    out["fetch_queue"] = []
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
