"""Built-in agent tools. Side effects only via ActionGate.execute."""

from __future__ import annotations

from typing import Any, Callable

from sqlalchemy.orm import Session

from prism.agents.memory import MemoryStore
from prism.agents.policy.rules import apply_overlay
from prism.agents.tools import ToolRegistry, ToolSpec
from prism.config import Settings, get_settings
from prism.db.models import Chunk, Document
from prism.rag.generate.grounding import check_citations, check_grounding
from prism.rag.retrieve import retrieve
from prism.rag.rewrite import rewrite as rewrite_query_fn


class ToolContext:
    """Shared mutable context for tool closures within one AgentRuntime."""

    def __init__(
        self,
        *,
        db: Session | None,
        settings: Settings | None = None,
        memory: MemoryStore | None = None,
        spawn_fn: Callable[..., Any] | None = None,
        status_fn: Callable[..., Any] | None = None,
        followup_fn: Callable[..., Any] | None = None,
        stop_fn: Callable[..., Any] | None = None,
        get_policy: Callable[[], Any] | None = None,
        set_policy: Callable[[Any], None] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.memory = memory or MemoryStore(db)
        self.spawn_fn = spawn_fn
        self.status_fn = status_fn
        self.followup_fn = followup_fn
        self.stop_fn = stop_fn
        self.get_policy = get_policy
        self.set_policy = set_policy
        self.last_hits: list[dict[str, Any]] = []
        self.notes: list[str] = []


def build_default_registry(ctx: ToolContext) -> ToolRegistry:
    reg = ToolRegistry()

    def search_kb(query: str, top_k: int | None = None) -> dict[str, Any]:
        if ctx.db is None:
            # Test/mock path
            hits = [
                {
                    "chunk_id": 1,
                    "document_id": 1,
                    "document_name": "mock.md",
                    "text": f"Mock evidence for: {query}",
                    "score": 0.9,
                    "section_path": None,
                    "page": None,
                }
            ]
            ctx.last_hits = hits
            return {"hits": hits, "retrieval_mode": "mock", "query": query}
        hits, mode = retrieve(ctx.db, query, top_k=top_k, settings=ctx.settings)
        packed = [
            {
                "chunk_id": h.chunk_id,
                "document_id": h.document_id,
                "document_name": h.document_name,
                "text": h.text,
                "score": h.score,
                "section_path": h.section_path,
                "page": h.page,
            }
            for h in hits
        ]
        ctx.last_hits = packed
        return {"hits": packed, "retrieval_mode": mode, "query": query}

    def get_document(document_id: int) -> dict[str, Any]:
        if ctx.db is None:
            return {"id": document_id, "name": "mock.md", "chunks": []}
        doc = ctx.db.get(Document, document_id)
        if doc is None:
            return {"error": "not_found", "id": document_id}
        chunks = (
            ctx.db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.id.asc())
            .limit(20)
            .all()
        )
        return {
            "id": doc.id,
            "name": doc.name,
            "space": doc.space,
            "status": doc.status,
            "chunks": [{"id": c.id, "text": c.text[:500], "section_path": c.section_path} for c in chunks],
        }

    def rewrite_query(query: str, strategy: str | None = None) -> dict[str, Any]:
        strat = strategy or ctx.settings.rewrite_strategy
        return rewrite_query_fn(query, strat, settings=ctx.settings)

    def verify_citations(answer: str, sources: list[str] | None = None) -> dict[str, Any]:
        src = sources
        if src is None:
            src = [h["text"] for h in ctx.last_hits]
        grounding = check_grounding(answer, src)
        citations = check_citations(answer, len(src))
        return {
            "grounding_ok": grounding.ok,
            "grounding_score": grounding.score,
            "citations_ok": citations.ok,
            "detail": grounding.detail,
        }

    def remember(layer: str, key: str, content: str) -> dict[str, Any]:
        mid = ctx.memory.remember(layer, key, content)
        return {"id": mid, "layer": layer, "key": key}

    def recall(query: str, layers: list[str] | None = None) -> dict[str, Any]:
        hits = ctx.memory.recall(query, layers=layers)
        return {
            "hits": [
                {"id": h.id, "layer": h.layer, "key": h.key, "content": h.content}
                for h in hits
            ]
        }

    def spawn_subagents(roles: list[str], objective: str) -> dict[str, Any]:
        if ctx.spawn_fn is None:
            return {"error": "spawn_unavailable"}
        return ctx.spawn_fn(roles=roles, objective=objective)

    def subagent_status(batch_id: str | None = None) -> dict[str, Any]:
        if ctx.status_fn is None:
            return {"error": "status_unavailable"}
        return ctx.status_fn(batch_id=batch_id)

    def subagent_followup(child_id: int, message: str) -> dict[str, Any]:
        if ctx.followup_fn is None:
            return {"error": "followup_unavailable"}
        return ctx.followup_fn(child_id=child_id, message=message)

    def subagent_stop(child_id: int) -> dict[str, Any]:
        if ctx.stop_fn is None:
            return {"error": "stop_unavailable"}
        return ctx.stop_fn(child_id=child_id)

    def propose_rule(tool: str, mode: str, rule_id: str | None = None) -> dict[str, Any]:
        if ctx.get_policy is None or ctx.set_policy is None:
            return {"accepted": False, "reason": "policy_unavailable"}
        bundle = ctx.get_policy()
        rid = rule_id or f"agent.propose.{tool}"
        new_bundle, ok, reason = apply_overlay(
            bundle, tool=tool, mode=mode, rule_id=rid, layer="agent"  # type: ignore[arg-type]
        )
        if ok:
            ctx.set_policy(new_bundle)
        return {"accepted": ok, "reason": reason, "tool": tool, "mode": mode}

    def web_search(query: str) -> dict[str, Any]:
        # Intentionally stubbed — policy default is ask; no external call in MVP.
        return {"error": "web_search_disabled", "query": query}

    specs = [
        ToolSpec(
            code="search_kb",
            category="retrieval",
            side_effect="read",
            schema={"query": "str", "top_k": "int?"},
            execute=search_kb,
        ),
        ToolSpec(
            code="get_document",
            category="retrieval",
            side_effect="read",
            schema={"document_id": "int"},
            execute=get_document,
        ),
        ToolSpec(
            code="rewrite_query",
            category="retrieval",
            side_effect="none",
            schema={"query": "str", "strategy": "str?"},
            execute=rewrite_query,
        ),
        ToolSpec(
            code="verify_citations",
            category="quality",
            side_effect="none",
            schema={"answer": "str", "sources": "list[str]?"},
            execute=verify_citations,
        ),
        ToolSpec(
            code="remember",
            category="memory",
            side_effect="write",
            schema={"layer": "str", "key": "str", "content": "str"},
            execute=remember,
        ),
        ToolSpec(
            code="recall",
            category="memory",
            side_effect="read",
            schema={"query": "str", "layers": "list[str]?"},
            execute=recall,
        ),
        ToolSpec(
            code="spawn_subagents",
            category="orchestration",
            side_effect="write",
            schema={"roles": "list[str]", "objective": "str"},
            execute=spawn_subagents,
        ),
        ToolSpec(
            code="subagent_status",
            category="orchestration",
            side_effect="read",
            schema={"batch_id": "str?"},
            execute=subagent_status,
        ),
        ToolSpec(
            code="subagent_followup",
            category="orchestration",
            side_effect="write",
            schema={"child_id": "int", "message": "str"},
            execute=subagent_followup,
        ),
        ToolSpec(
            code="subagent_stop",
            category="orchestration",
            side_effect="write",
            schema={"child_id": "int"},
            execute=subagent_stop,
        ),
        ToolSpec(
            code="propose_rule",
            category="policy",
            side_effect="write",
            schema={"tool": "str", "mode": "str", "rule_id": "str?"},
            execute=propose_rule,
        ),
        ToolSpec(
            code="web_search",
            category="external",
            side_effect="external",
            schema={"query": "str"},
            execute=web_search,
        ),
    ]
    for spec in specs:
        reg.register(spec)
    return reg
