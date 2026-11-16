"""Retrieval entrypoint: dense / lexical / hybrid + optional rewrite & rerank."""

from __future__ import annotations

import logging
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from prism.config import Settings, get_settings
from prism.db.models import Chunk, Document
from prism.observability import span
from prism.rag.embeddings import get_embedder
from prism.rag.index.bm25_store import BM25Store
from prism.rag.rerank import rerank as cross_rerank
from prism.rag.retrieve.fusion import rrf_fuse, weighted_fuse
from prism.rag.retrieve.packer import pack_context
from prism.rag.retrieve.spaces import apply_space_weights, space_weights_from_settings
from prism.rag.retrieve.types import RetrievedChunk
from prism.rag.rewrite import rewrite as rewrite_query

logger = logging.getLogger(__name__)


def retrieve(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Return packed chunks and retrieval_mode string (includes fallbacks)."""
    cfg = settings or get_settings()
    k = top_k or cfg.top_k
    mode = cfg.retrieval_mode
    trace_parts = [mode]

    with span(
        "rag.retrieve",
        attributes={"mode": mode, "top_k": k, "rewrite": cfg.rewrite_strategy},
    ):
        return _retrieve_inner(
            db, query, top_k=k, settings=cfg, history=history, trace_parts=trace_parts
        )


def _retrieve_inner(
    db: Session,
    query: str,
    *,
    top_k: int,
    settings: Settings,
    history: list[dict] | None,
    trace_parts: list[str],
) -> tuple[list[RetrievedChunk], str]:
    cfg = settings
    k = top_k
    mode = cfg.retrieval_mode

    rw = rewrite_query(query, cfg.rewrite_strategy, history=history, settings=cfg)
    semantic_q = rw["semantic_query"]
    lexical_q = rw.get("lexical_query") or query
    multi_queries: list[str] = rw.get("queries") or [semantic_q]
    if cfg.rewrite_strategy != "R0":
        trace_parts.append(f"rewrite:{cfg.rewrite_strategy}")

    # R5 may bump top_k / prefer hybrid
    if cfg.rewrite_strategy == "R5" and isinstance(rw.get("plan"), dict):
        plan = rw["plan"]
        k = max(k, int(plan.get("top_k") or k))
        prefer = plan.get("prefer")
        if prefer in ("dense", "lexical", "hybrid"):
            mode = prefer
            trace_parts[0] = mode

    candidate_k = max(k * 4, cfg.rerank_candidates if cfg.rerank_enabled else 20)

    hits: list[RetrievedChunk] = []
    try:
        if mode == "dense":
            if len(multi_queries) > 1:
                hits = _multi_dense_rrf(db, multi_queries, candidate_k=candidate_k, settings=cfg)
                trace_parts.append("multi_query_rrf")
            else:
                hits = _dense_search(db, semantic_q, candidate_k=candidate_k, settings=cfg)
        elif mode == "lexical":
            hits = _lexical_search(db, lexical_q, candidate_k=candidate_k, settings=cfg)
            if not hits:
                hits = _dense_search(db, semantic_q, candidate_k=candidate_k, settings=cfg)
                trace_parts.append("lexical->dense_fallback")
        elif mode == "hybrid":
            hits = _hybrid_search(
                db,
                semantic_q=semantic_q,
                lexical_q=lexical_q,
                candidate_k=candidate_k,
                settings=cfg,
            )
            if not hits:
                hits = _dense_search(db, semantic_q, candidate_k=candidate_k, settings=cfg)
                trace_parts.append("hybrid->dense_fallback")
        else:
            raise ValueError(f"Unknown retrieval_mode: {mode}")
    except Exception:
        logger.exception("Primary retrieval failed; dense fallback")
        hits = _dense_search(db, semantic_q, candidate_k=candidate_k, settings=cfg)
        trace_parts.append("error->dense_fallback")

    if cfg.rerank_enabled and hits:
        t0 = perf_counter()
        order = cross_rerank(
            semantic_q,
            [h.text for h in hits],
            top_n=min(k, len(hits)),
            model_name=cfg.rerank_model,
        )
        reranked = [hits[i] for i, _ in order]
        # attach rerank score into a copy-friendly way via meta not available — keep order
        hits = reranked
        trace_parts.append(f"rerank:{int((perf_counter() - t0) * 1000)}ms")

    # C7: expand child → parent text for generation context when present
    hits = [_maybe_expand_parent(h) for h in hits]

    if cfg.space_weights_enabled:
        hits = apply_space_weights(hits, space_weights_from_settings(cfg))
        trace_parts.append("space_weights")

    with span(
        "rag.pack",
        attributes={"max_chunks": cfg.pack_max_chunks, "candidate_n": len(hits)},
    ):
        packed = pack_context(
            hits,
            max_chunks=cfg.pack_max_chunks,
            max_tokens=cfg.pack_max_tokens,
            max_per_doc=cfg.pack_max_per_doc,
            min_score=cfg.pack_min_score,
        )
    return packed[:k], "+".join(trace_parts)


def _maybe_expand_parent(h: RetrievedChunk) -> RetrievedChunk:
    """C7 small-to-big: replace child text with parent_text for the generator."""
    meta = h.meta or {}
    parent = meta.get("parent_text")
    if not parent or parent == h.text:
        return h
    return RetrievedChunk(
        chunk_id=h.chunk_id,
        document_id=h.document_id,
        document_name=h.document_name,
        text=str(parent),
        score=h.score,
        section_path=h.section_path,
        page=h.page,
        space=h.space,
        doc_key=h.doc_key,
        meta={**meta, "child_text": h.text, "expanded": True},
    )


def _hybrid_search(
    db: Session,
    *,
    semantic_q: str,
    lexical_q: str,
    candidate_k: int,
    settings: Settings,
) -> list[RetrievedChunk]:
    dense_hits = _dense_search(db, semantic_q, candidate_k=candidate_k, settings=settings)
    lex_hits = _lexical_search(db, lexical_q, candidate_k=candidate_k, settings=settings)

    by_id: dict[int, RetrievedChunk] = {h.chunk_id: h for h in dense_hits}
    for h in lex_hits:
        by_id.setdefault(h.chunk_id, h)

    dense_ids = [str(h.chunk_id) for h in dense_hits]
    lex_ids = [str(h.chunk_id) for h in lex_hits]

    if settings.hybrid_fusion == "rrf":
        fused = rrf_fuse([dense_ids, lex_ids], k=settings.rrf_k)
    else:
        dense_scores = {str(h.chunk_id): h.score for h in dense_hits}
        lex_scores = {str(h.chunk_id): h.score for h in lex_hits}
        fused = weighted_fuse([dense_scores, lex_scores], alpha=settings.hybrid_alpha)

    out: list[RetrievedChunk] = []
    for cid, score in fused[:candidate_k]:
        hit = by_id.get(int(cid))
        if hit is None:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_name=hit.document_name,
                text=hit.text,
                score=float(score),
                section_path=hit.section_path,
                page=hit.page,
                space=hit.space,
                doc_key=hit.doc_key,
                meta=hit.meta,
            )
        )
    return out


def _multi_dense_rrf(
    db: Session,
    queries: list[str],
    *,
    candidate_k: int,
    settings: Settings,
) -> list[RetrievedChunk]:
    rankings: list[list[str]] = []
    by_id: dict[int, RetrievedChunk] = {}
    for q in queries:
        hits = _dense_search(db, q, candidate_k=candidate_k, settings=settings)
        rankings.append([str(h.chunk_id) for h in hits])
        for h in hits:
            by_id.setdefault(h.chunk_id, h)
    fused = rrf_fuse(rankings, k=settings.rrf_k)
    out: list[RetrievedChunk] = []
    for cid, score in fused[:candidate_k]:
        hit = by_id.get(int(cid))
        if not hit:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_name=hit.document_name,
                text=hit.text,
                score=float(score),
                section_path=hit.section_path,
                page=hit.page,
                space=hit.space,
                doc_key=hit.doc_key,
                meta=hit.meta,
            )
        )
    return out


def _lexical_search(
    db: Session,
    query: str,
    *,
    candidate_k: int,
    settings: Settings,
) -> list[RetrievedChunk]:
    store = BM25Store(settings.bm25_dir)
    pairs = store.search(query, top_k=candidate_k)
    if not pairs:
        return []
    ids = [int(cid) for cid, _ in pairs]
    score_map = {int(cid): float(sc) for cid, sc in pairs}
    rows = db.execute(
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.text,
            Chunk.section_path,
            Chunk.page,
            Chunk.space,
            Chunk.meta.label("chunk_meta"),
            Document.name.label("document_name"),
            Document.meta.label("document_meta"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(ids))
    ).mappings()
    by_id = {int(r["id"]): r for r in rows}
    out: list[RetrievedChunk] = []
    for cid in ids:
        r = by_id.get(cid)
        if not r:
            continue
        doc_meta = r["document_meta"] or {}
        chunk_meta = r["chunk_meta"] or {}
        doc_key = doc_meta.get("doc_key") or chunk_meta.get("doc_key")
        if isinstance(doc_key, str) and doc_key.startswith("demo_"):
            doc_key = doc_key[len("demo_") :]
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                document_id=int(r["document_id"]),
                document_name=r["document_name"],
                text=r["text"],
                score=score_map.get(cid, 0.0),
                section_path=r["section_path"],
                page=r["page"],
                space=r["space"],
                doc_key=doc_key,
                meta=dict(chunk_meta) if chunk_meta else None,
            )
        )
    return out


def _dense_search(
    db: Session,
    query: str,
    *,
    candidate_k: int,
    settings: Settings,
) -> list[RetrievedChunk]:
    embedder = get_embedder(settings.embedding_model)
    qvec = embedder.embed_query(query)
    distance = Chunk.embedding.cosine_distance(qvec)

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.text,
            Chunk.section_path,
            Chunk.page,
            Chunk.space,
            Chunk.meta.label("chunk_meta"),
            Document.name.label("document_name"),
            Document.meta.label("document_meta"),
            distance.label("distance"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.embedding.is_not(None),
            Document.status == "indexed",
            Chunk.embedding_model == embedder.model_name,
        )
        .order_by(distance)
        .limit(candidate_k)
    )

    rows = db.execute(stmt).mappings()
    out: list[RetrievedChunk] = []
    for r in rows:
        doc_meta = r["document_meta"] or {}
        chunk_meta = r["chunk_meta"] or {}
        doc_key = doc_meta.get("doc_key") or chunk_meta.get("doc_key")
        if not doc_key and r["document_name"]:
            doc_key = str(r["document_name"]).rsplit(".", 1)[0]
        if isinstance(doc_key, str) and doc_key.startswith("demo_"):
            doc_key = doc_key[len("demo_") :]
        out.append(
            RetrievedChunk(
                chunk_id=int(r["id"]),
                document_id=int(r["document_id"]),
                document_name=r["document_name"],
                text=r["text"],
                score=max(0.0, min(1.0, 1.0 - float(r["distance"]) / 2.0)),
                section_path=r["section_path"],
                page=r["page"],
                space=r["space"],
                doc_key=doc_key,
                meta=dict(chunk_meta) if chunk_meta else None,
            )
        )
    return out


def delete_document_chunks(db: Session, document_id: int) -> int:
    chunks = db.scalars(select(Chunk).where(Chunk.document_id == document_id)).all()
    n = len(chunks)
    for ch in chunks:
        db.delete(ch)
    return n
