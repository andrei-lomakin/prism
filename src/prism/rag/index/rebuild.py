"""Rebuild BM25 from all indexed chunks in Postgres."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from prism.config import get_settings
from prism.db.models import Chunk, Document
from prism.rag.index.bm25_store import BM25Store

logger = logging.getLogger(__name__)


def rebuild_bm25(db: Session, *, collection: str = "default") -> int:
    settings = get_settings()
    rows = db.execute(
        select(Chunk.id, Chunk.text)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == "indexed", Chunk.embedding.is_not(None))
        .order_by(Chunk.id)
    ).all()
    ids = [str(r[0]) for r in rows]
    texts = [r[1] for r in rows]
    store = BM25Store(settings.bm25_dir, collection=collection)
    store.build(ids, texts)
    logger.info("BM25 rebuilt: %s chunks", len(ids))
    return len(ids)
