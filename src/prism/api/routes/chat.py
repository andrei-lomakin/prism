"""Search + RAG chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from prism.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    SearchHit,
    SearchRequest,
    SearchResponse,
    UsageOut,
)
from prism.api.settings_store import settings_as_config
from prism.db.session import get_db
from prism.llm import LLMGateway
from prism.observability import span, trace_buffer
from prism.rag.generate import SYSTEM_PROMPT, build_rag_prompt, check_grounding
from prism.rag.retrieve import retrieve

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/search", response_model=SearchResponse)
def api_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    cfg = settings_as_config(db)
    tid = trace_buffer.start_trace()
    try:
        with span("api.search", attributes={"trace_id": tid}):
            hits, mode = retrieve(db, payload.query, top_k=payload.top_k, settings=cfg)
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

    if payload.mode == "agent":
        from prism.agents import AgentRuntime

        if payload.top_k is not None:
            cfg.top_k = payload.top_k
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
                    chunk_id=int(c.get("chunk_id") or 0),
                    doc_id=int(c.get("doc_id") or 0),
                    title=str(c.get("title") or ""),
                    section_path=c.get("section_path"),
                    page=c.get("page"),
                    score=round(float(c.get("score") or 0.0), 4),
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
            hits, mode = retrieve(db, payload.query, top_k=payload.top_k, settings=cfg)
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
            gateway = LLMGateway(cfg)
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
        usage=UsageOut(
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        ),
        trace_id=tid,
    )
