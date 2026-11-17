"""Built-in agent tools. Side effects only via ActionGate.execute."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from prism.agents.memory import MemoryStore
from prism.agents.memory.distill import distill_run_to_memory
from prism.agents.policy.rules import apply_overlay
from prism.agents.tools import ToolRegistry, ToolSpec
from prism.config import Settings, get_settings
from prism.db.models import Chunk, Document
from prism.net.safe_http import UnsafeUrlError, assert_public_http_url, safe_get
from prism.rag.generate.grounding import check_citations, check_grounding
from prism.rag.retrieve import retrieve
from prism.rag.rewrite import rewrite as rewrite_query_fn

logger = logging.getLogger(__name__)


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
        self.dialog_history: list[dict[str, str]] = []


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
        settings = ctx.settings
        history = list(ctx.dialog_history or [])
        if history and getattr(settings, "rewrite_strategy", "R0") == "R0":
            settings = settings.model_copy(update={"rewrite_strategy": "R1"})
        hits, mode = retrieve(
            ctx.db,
            query,
            top_k=top_k,
            settings=settings,
            history=history or None,
            space=ctx.settings.space_filter,
        )
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
            "chunks": [
                {"id": c.id, "text": c.text[:500], "section_path": c.section_path} for c in chunks
            ],
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
            "citations_ok": citations.ok and not citations.missing,
            "cited": grounding.cited,
            "missing": grounding.missing,
            "unsupported_numbers": grounding.unsupported_numbers,
            "unsupported_names": grounding.unsupported_names,
        }

    def remember(layer: str, key: str, content: str) -> dict[str, Any]:
        mid = ctx.memory.remember(layer, key, content)
        return {"id": mid, "layer": layer, "key": key}

    def recall(query: str, layers: list[str] | None = None) -> dict[str, Any]:
        hits = ctx.memory.recall(query, layers=layers)
        return {
            "hits": [
                {"id": h.id, "layer": h.layer, "key": h.key, "content": h.content} for h in hits
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
            bundle,
            tool=tool,
            mode=mode,
            rule_id=rid,
            layer="agent",  # type: ignore[arg-type]
        )
        if ok:
            ctx.set_policy(new_bundle)
        return {"accepted": ok, "reason": reason, "tool": tool, "mode": mode}

    def web_search(query: str) -> dict[str, Any]:
        """External search (ask-gated). Mock mode is labeled; live mode never invents hits."""
        q = (query or "").strip()
        if getattr(ctx.settings, "web_search_mock", True):
            mock = [
                {
                    "title": f"Overview: {q[:80]}",
                    "url": f"https://example.com/search?q={quote_plus(q)}",
                    "snippet": f"Mock web snippet about {q}. Offline demo, not a live search.",
                    "origin": "mock",
                    "provider": "mock",
                },
                {
                    "title": f"Comparison notes: {q[:60]}",
                    "url": "https://example.com/compare",
                    "snippet": "Side-by-side criteria, trade-offs, latency vs quality.",
                    "origin": "mock",
                    "provider": "mock",
                },
            ]
            return {
                "query": q,
                "results": mock,
                "provider": "mock",
                "origin": "mock",
                "ok": True,
                "status": "ok",
            }
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(q)}"
        try:
            body, _ctype, _final = safe_get(
                url,
                timeout=12.0,
                max_bytes=2_000_000,
                headers={"User-Agent": "PrismResearchBot/0.1"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_search failed: %s", exc)
            return {
                "query": q,
                "results": [],
                "provider": "duckduckgo_lite",
                "origin": "live",
                "ok": False,
                "status": "error",
                "error": str(exc),
            }
        try:
            soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "lxml")
            results: list[dict[str, Any]] = []
            for anchor in soup.select("a.result-link")[:5]:
                href = anchor.get("href") or ""
                title = anchor.get_text(" ", strip=True)
                if href and title:
                    results.append(
                        {
                            "title": title,
                            "url": href,
                            "snippet": title,
                            "origin": "live",
                            "provider": "duckduckgo_lite",
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_search parse failed: %s", exc)
            return {
                "query": q,
                "results": [],
                "provider": "duckduckgo_lite",
                "origin": "live",
                "ok": False,
                "status": "error",
                "error": f"parse: {exc}",
            }
        if not results:
            return {
                "query": q,
                "results": [],
                "provider": "duckduckgo_lite",
                "origin": "live",
                "ok": True,
                "status": "empty",
                "error": None,
            }
        return {
            "query": q,
            "results": results,
            "provider": "duckduckgo_lite",
            "origin": "live",
            "ok": True,
            "status": "ok",
        }

    def fetch_page(url: str) -> dict[str, Any]:
        mock = bool(getattr(ctx.settings, "web_search_mock", True))
        try:
            assert_public_http_url(url, resolve=not mock)
        except UnsafeUrlError as exc:
            return {
                "url": url,
                "ok": False,
                "error": str(exc),
                "origin": "blocked",
                "status": "error",
                "text": "",
            }
        if mock:
            text = f"Mock page body for {url}. Offline demo, not a fetched page."
            return {
                "url": url,
                "title": "Mock page",
                "text": text,
                "chars": len(text),
                "origin": "mock",
                "provider": "mock",
                "ok": True,
                "status": "ok",
            }
        try:
            body, _ctype, final = safe_get(
                url,
                timeout=15.0,
                max_bytes=1_000_000,
                headers={"User-Agent": "PrismResearchBot/0.1"},
            )
            soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = (soup.title.get_text(strip=True) if soup.title else final)[:200]
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:4000]
            return {
                "url": final,
                "title": title,
                "text": text,
                "chars": len(text),
                "origin": "live",
                "provider": "live",
                "ok": True,
                "status": "ok",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "url": url,
                "ok": False,
                "error": str(exc),
                "origin": "live",
                "status": "error",
                "text": "",
            }

    def build_comparison_table(
        objective: str,
        kb_hits: list[dict[str, Any]] | None = None,
        web_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kb_hits = kb_hits or []
        web_results = web_results or []
        rows: list[dict[str, str]] = []
        for i, h in enumerate(kb_hits[:4], start=1):
            rows.append(
                {
                    "aspect": f"KB[{i}] {h.get('document_name') or 'doc'}",
                    "evidence": (h.get("text") or "")[:180],
                    "source": "kb",
                }
            )
        for i, w in enumerate(web_results[:4], start=1):
            rows.append(
                {
                    "aspect": f"Web[{i}] {w.get('title') or 'page'}",
                    "evidence": (w.get("snippet") or w.get("text") or "")[:180],
                    "source": "web",
                }
            )
        if not rows:
            rows = [
                {
                    "aspect": "insufficient evidence",
                    "evidence": f"No rows for: {objective}",
                    "source": "none",
                }
            ]
        md = "| Aspect | Evidence | Source |\n|---|---|---|\n" + "\n".join(
            f"| {r['aspect'].replace('|', '/')} | {r['evidence'].replace('|', '/')} | {r['source']} |"
            for r in rows
        )
        return {"objective": objective, "rows": rows, "markdown": md}

    def judge(answer: str, question: str, sources: list[str] | None = None) -> dict[str, Any]:
        sources = sources or []
        g = check_grounding(answer, sources)
        # Soft completeness: answer should mention at least one contentful token from question
        tokens = [t for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", question.lower())]
        covered = sum(1 for t in tokens if t in (answer or "").lower())
        completeness = covered / max(1, len(tokens))
        ok = g.ok and completeness >= 0.2
        return {
            "ok": ok,
            "grounding_ok": g.ok,
            "completeness": round(completeness, 3),
            "unsupported_numbers": g.unsupported_numbers,
            "needs_revision": not ok,
        }

    def distill_run(
        run_id: str,
        objective: str,
        answer: str,
        citations: list[dict[str, Any]] | None = None,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return distill_run_to_memory(
            ctx.memory,
            run_id=run_id,
            objective=objective,
            answer=answer,
            citations=citations,
            plan=plan,
        )

    def python_sandbox(code: str) -> dict[str, Any]:
        """Restricted stub: no exec. Parses simple `print(expr)` arithmetic only."""
        code = (code or "").strip()
        m = re.fullmatch(r"print\((.+)\)", code)
        if not m:
            return {
                "ok": False,
                "error": "sandbox_stub: only print(<arithmetic>) allowed",
                "stdout": "",
            }
        expr = m.group(1)
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return {"ok": False, "error": "sandbox_stub: disallowed tokens", "stdout": ""}
        try:
            val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — intentional tiny arith
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "stdout": ""}
        return {"ok": True, "stdout": str(val), "expr": expr}

    def memory_rollback(entry_id: int) -> dict[str, Any]:
        return ctx.memory.rollback(int(entry_id))

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
        ToolSpec(
            code="fetch_page",
            category="external",
            side_effect="external",
            schema={"url": "str"},
            execute=fetch_page,
        ),
        ToolSpec(
            code="build_comparison_table",
            category="analysis",
            side_effect="none",
            schema={"objective": "str", "kb_hits": "list?", "web_results": "list?"},
            execute=build_comparison_table,
        ),
        ToolSpec(
            code="judge",
            category="quality",
            side_effect="none",
            schema={"answer": "str", "question": "str", "sources": "list[str]?"},
            execute=judge,
        ),
        ToolSpec(
            code="distill_run",
            category="memory",
            side_effect="write",
            schema={
                "run_id": "str",
                "objective": "str",
                "answer": "str",
                "citations": "list?",
                "plan": "dict?",
            },
            execute=distill_run,
        ),
        ToolSpec(
            code="python_sandbox",
            category="analysis",
            side_effect="none",
            schema={"code": "str"},
            execute=python_sandbox,
        ),
        ToolSpec(
            code="memory_rollback",
            category="memory",
            side_effect="write",
            schema={"entry_id": "int"},
            execute=memory_rollback,
        ),
    ]
    for spec in specs:
        reg.register(spec)
    return reg
