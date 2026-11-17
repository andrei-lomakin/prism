"""Search + RAG chat endpoints."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from prism.api.auth import require_user
from prism.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    GroundingOut,
    SearchHit,
    SearchRequest,
    SearchResponse,
    UsageOut,
)
from prism.rag.generate.grounding import GroundingReport
from prism.api.settings_store import settings_as_config
from prism.db.session import get_db, get_session_factory
from prism.llm import gateway_for
from prism.observability import span, trace_buffer
from prism.rag.generate import SYSTEM_PROMPT, build_rag_prompt, check_grounding
from prism.rag.retrieve import retrieve

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(require_user)])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _grounding_out(report: GroundingReport) -> GroundingOut:
    return GroundingOut(
        ok=report.ok,
        cited=list(report.cited),
        missing=list(report.missing),
        unsupported_numbers=list(report.unsupported_numbers),
        unsupported_names=list(report.unsupported_names),
        requires_citation=bool(report.requires_citation),
    )


def _grounding_dict(report: GroundingReport) -> dict[str, Any]:
    return _grounding_out(report).model_dump()


@router.post("/search", response_model=SearchResponse)
def api_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    cfg = settings_as_config(db)
    tid = trace_buffer.start_trace()
    try:
        with span("api.search", attributes={"trace_id": tid}):
            hits, mode = retrieve(
                db,
                payload.query,
                top_k=payload.top_k,
                settings=cfg,
                space=payload.space,
            )
    except Exception as exc:
        raise HTTPException(503, f"Search failed: {exc}") from exc
    return SearchResponse(
        query=payload.query,
        retrieval_mode=mode,
        results=[
            SearchHit(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                document_name=h.document_name,
                text=h.text,
                score=round(h.score, 4),
                section_path=h.section_path,
                page=h.page,
            )
            for h in hits
        ],
    )


@router.post("/chat", response_model=ChatResponse)
def api_chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    cfg = settings_as_config(db)

    tid = trace_buffer.start_trace()

    if payload.top_k is not None:
        cfg.top_k = payload.top_k
    cfg.space_filter = None if payload.space == "all" else payload.space

    if payload.mode == "agent":
        from prism.agents import AgentRuntime

        try:
            with span("api.chat.agent", attributes={"trace_id": tid}):
                runtime = AgentRuntime(db, settings=cfg, persist=True)
                answer = runtime.handle(payload.query, role="orchestrator")
        except Exception as exc:
            raise HTTPException(503, f"Agent run failed: {exc}") from exc
        return ChatResponse(
            query=payload.query,
            answer=answer.answer,
            model=answer.model,
            citations=[
                Citation(
                    n=c["n"],
                    chunk_id=c.get("chunk_id"),
                    doc_id=c.get("doc_id"),
                    title=str(c.get("title") or ""),
                    section_path=c.get("section_path"),
                    page=c.get("page"),
                    score=round(float(c.get("score") or 0.0), 4),
                    text=(c.get("text") or None),
                    url=c.get("url"),
                    kind=c.get("kind") or ("web" if c.get("url") else "kb"),
                )
                for c in answer.citations
            ],
            retrieval_mode=answer.retrieval_mode,
            grounding_ok=answer.grounding_ok,
            usage=UsageOut(
                tokens_in=answer.tokens_in,
                tokens_out=answer.tokens_out,
                cost_usd=answer.cost_usd,
                latency_ms=answer.latency_ms,
            ),
            trace_id=str(answer.run_id),
        )

    try:
        with span("api.chat.rag", attributes={"trace_id": tid}):
            hits, mode = retrieve(
                db,
                payload.query,
                top_k=payload.top_k,
                settings=cfg,
                space=None if payload.space == "all" else payload.space,
            )
            contexts = [
                {
                    "text": h.text,
                    "document_name": h.document_name,
                    "score": h.score,
                    "section_path": h.section_path,
                }
                for h in hits
            ]
            user_prompt = build_rag_prompt(payload.query, contexts)
            gateway = gateway_for(cfg)
            result = gateway.chat(user_message=user_prompt, system_prompt=SYSTEM_PROMPT)
    except Exception as exc:
        raise HTTPException(503, f"RAG chat failed: {exc}") from exc

    grounding = check_grounding(result.text, [h.text for h in hits])
    citations = [
        Citation(
            n=i,
            chunk_id=h.chunk_id,
            doc_id=h.document_id,
            title=h.document_name,
            section_path=h.section_path,
            page=h.page,
            score=round(h.score, 4),
            text=h.text,
        )
        for i, h in enumerate(hits, start=1)
    ]
    return ChatResponse(
        query=payload.query,
        answer=result.text,
        model=result.model,
        citations=citations,
        retrieval_mode=mode,
        grounding_ok=grounding.ok,
        grounding=_grounding_out(grounding),
        usage=UsageOut(
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        ),
        trace_id=tid,
    )


@router.post("/chat/stream")
def api_chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """SSE: status/retrieval/plan/step + live LLM ``token`` deltas (RAG).

    RAG uses provider ``stream: true``. Agent streams plan/steps while running;
    the final answer is one shot after ``handle()`` (multi-call orchestration).
    """
    cfg = settings_as_config(db)
    tid = trace_buffer.start_trace()

    def gen() -> Iterator[str]:
        yield _sse("start", {"trace_id": tid, "mode": payload.mode})
        try:
            if payload.top_k is not None:
                cfg.top_k = payload.top_k
            cfg.space_filter = None if payload.space == "all" else payload.space
            if payload.mode == "agent":
                from prism.agents import AgentRuntime

                # Run agent in a side thread with its own DB session so we can
                # SSE-emit steps while LLM/tool calls are still in progress.
                # (Request-scoped `db` is not thread-safe.)
                events: queue.Queue[tuple[str, Any]] = queue.Queue()

                def _worker() -> None:
                    session = get_session_factory()()
                    try:

                        def _on_step(rec: dict[str, Any]) -> None:
                            events.put(("step", rec))

                        runtime = AgentRuntime(
                            session,
                            settings=cfg,
                            persist=True,
                            on_step=_on_step,
                        )
                        events.put(
                            (
                                "status",
                                {
                                    "text": "Оцениваем сложность запроса и строим план…",
                                },
                            )
                        )
                        ans = runtime.handle(payload.query, role="orchestrator")
                        events.put(("done", ans))
                    except Exception as exc:  # noqa: BLE001
                        events.put(("error", exc))
                    finally:
                        session.close()

                threading.Thread(target=_worker, daemon=True).start()
                yield _sse(
                    "status",
                    {
                        "text": "Агент запущен: оценка сложности → инструменты → ответ "
                        "(ожидание LLM может быть долгим)…",
                    },
                )
                answer = None
                while True:
                    kind, payload_evt = events.get()
                    if kind == "status":
                        yield _sse("status", payload_evt)
                    elif kind == "step":
                        step = payload_evt
                        if step.get("kind") == "plan":
                            yield _sse("plan", step.get("output") or {})
                            yield _sse(
                                "status",
                                {"text": "План готов, выполняем инструменты…"},
                            )
                        yield _sse("step", step)
                        title = step.get("kind") or "шаг"
                        yield _sse(
                            "status",
                            {"text": f"Агент: шаг {step.get('n', '?')} ({title})…"},
                        )
                    elif kind == "done":
                        answer = payload_evt
                        break
                    elif kind == "error":
                        raise payload_evt

                assert answer is not None
                text = answer.answer or ""
                if text:
                    yield _sse("status", {"text": "Ответ готов"})
                    yield _sse("token", {"text": text})
                yield _sse(
                    "final",
                    {
                        "answer": answer.answer,
                        "model": answer.model,
                        "citations": answer.citations,
                        "retrieval_mode": answer.retrieval_mode,
                        "grounding_ok": answer.grounding_ok,
                        "grounding": {
                            "ok": bool(answer.grounding_ok),
                            "cited": [],
                            "missing": [],
                            "unsupported_numbers": [],
                            "unsupported_names": [],
                            "requires_citation": False,
                        },
                        "trace_id": str(answer.run_id),
                        "state": answer.state,
                        "usage": answer.usage,
                        "provisional_until_grounding": False,
                    },
                )
            else:
                yield _sse("status", {"text": "Ищем релевантные фрагменты в базе…"})
                hits, mode = retrieve(
                    db,
                    payload.query,
                    top_k=payload.top_k,
                    settings=cfg,
                    space=None if payload.space == "all" else payload.space,
                )
                yield _sse(
                    "retrieval",
                    {"mode": mode, "n": len(hits), "docs": [h.document_name for h in hits[:5]]},
                )
                yield _sse("status", {"text": "Генерируем ответ по найденному контексту…"})
                contexts = [
                    {
                        "text": h.text,
                        "document_name": h.document_name,
                        "score": h.score,
                        "section_path": h.section_path,
                    }
                    for h in hits
                ]
                user_prompt = build_rag_prompt(payload.query, contexts)
                gateway = gateway_for(cfg)
                stream = gateway.chat_stream(
                    user_message=user_prompt,
                    system_prompt=SYSTEM_PROMPT,
                )
                result = None
                while True:
                    try:
                        delta = next(stream)
                    except StopIteration as stop:
                        result = stop.value
                        break
                    if delta:
                        yield _sse("token", {"text": delta})
                if result is None:
                    raise RuntimeError("LLM stream returned no result")
                grounding = check_grounding(result.text, [h.text for h in hits])
                yield _sse(
                    "final",
                    {
                        "answer": result.text,
                        "model": result.model,
                        "citations": [
                            {
                                "n": i,
                                "chunk_id": h.chunk_id,
                                "doc_id": h.document_id,
                                "title": h.document_name,
                                "section_path": h.section_path,
                                "page": h.page,
                                "score": round(h.score, 4),
                                "text": h.text,
                            }
                            for i, h in enumerate(hits, start=1)
                        ],
                        "retrieval_mode": mode,
                        "grounding_ok": grounding.ok,
                        "grounding": _grounding_dict(grounding),
                        "trace_id": tid,
                        "usage": {
                            "tokens_in": result.tokens_in,
                            "tokens_out": result.tokens_out,
                            "cost_usd": result.cost_usd,
                            "latency_ms": result.latency_ms,
                        },
                        # Stream text was provisional; grounding ran after generation.
                        "provisional_until_grounding": True,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"detail": str(exc)})
        yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream")
