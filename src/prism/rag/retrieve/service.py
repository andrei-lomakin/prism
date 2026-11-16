"""Retrieval entrypoint.

Stage 0: dense (pgvector cosine). Lexical / hybrid / fallbacks arrive in stage 3.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from prism.config import Settings, get_settings
from prism.db.models import Chunk, Document
from prism.rag.embeddings import get_embedder
from prism.rag.retrieve.packer import pack_context
from prism.rag.retrieve.types import RetrievedChunk


def retrieve(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Return packed chunks and retrieval_mode used."""
    cfg = settings or get_settings()
    k = top_k or cfg.top_k
    mode = cfg.retrieval_mode

    if mode == "dense":
        hits = _dense_search(db, query, candidate_k=max(k * 4, 20), settings=cfg)
    elif mode in ("lexical", "hybrid"):
        # Stage 3 — graceful fallback so API stays usable.
        hits = _dense_search(db, query, candidate_k=max(k * 4, 20), settings=cfg)
        mode = f"{mode}->dense_fallback"
    else:
        raise ValueError(f"Unknown retrieval_mode: {mode}")

    packed = pack_context(
        hits,
        max_chunks=cfg.pack_max_chunks,
        max_tokens=cfg.pack_max_tokens,
        max_per_doc=cfg.pack_max_per_doc,
        min_score=cfg.pack_min_score,
    )
    return packed[:k], mode


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
            Document.name.label("document_name"),
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
    return [
        RetrievedChunk(
            chunk_id=int(r["id"]),
            document_id=int(r["document_id"]),
            document_name=r["document_name"],
            text=r["text"],
            # cosine_distance ∈ [0, 2] → score ∈ [0, 1]
            score=max(0.0, min(1.0, 1.0 - float(r["distance"]) / 2.0)),
            section_path=r["section_path"],
            page=r["page"],
            space=r["space"],
        )
        for r in rows
    ]


def delete_document_chunks(db: Session, document_id: int) -> int:
    chunks = db.scalars(select(Chunk).where(Chunk.document_id == document_id)).all()
    n = len(chunks)
    for ch in chunks:
        db.delete(ch)
    return n
