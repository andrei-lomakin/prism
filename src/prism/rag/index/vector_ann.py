"""pgvector ANN index helpers: exact seq-scan vs HNSW."""

from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

VectorSearchMode = Literal["exact", "hnsw"]
HNSW_INDEX = "chunks_embedding_hnsw_cosine_idx"


def ensure_vector_index(
    engine: Engine, mode: VectorSearchMode, *, ef_construction: int = 64
) -> str:
    """Create or drop HNSW index on chunks.embedding to match the chosen mode."""
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_indexes WHERE indexname = :n"),
            {"n": HNSW_INDEX},
        ).scalar()
        if mode == "hnsw":
            if exists:
                return "hnsw:exists"
            logger.info("Creating HNSW index %s (may take a while)…", HNSW_INDEX)
            ef = max(4, int(ef_construction))
            conn.execute(
                text(
                    f"""
                    CREATE INDEX {HNSW_INDEX}
                    ON chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = {ef})
                    """
                )
            )
            return "hnsw:created"
        if exists:
            logger.info("Dropping HNSW index %s (exact mode)", HNSW_INDEX)
            conn.execute(text(f"DROP INDEX IF EXISTS {HNSW_INDEX}"))
            return "exact:dropped"
        return "exact:ok"


def apply_vector_search_hints(db: Session, mode: VectorSearchMode, *, ef_search: int = 40) -> None:
    """Tune planner for this transaction: force seq-scan (exact) or HNSW ef_search."""
    if mode == "exact":
        # Prefer sequential scan so cosine ORDER BY is exact even if HNSW exists.
        db.execute(text("SET LOCAL enable_indexscan = off"))
        db.execute(text("SET LOCAL enable_bitmapscan = off"))
        return
    # HNSW path — allow index; raise ef_search for better recall.
    db.execute(text("SET LOCAL enable_indexscan = on"))
    try:
        db.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    except Exception:
        # Older pgvector without GUC — index still used if present.
        logger.debug("hnsw.ef_search GUC unavailable", exc_info=True)
