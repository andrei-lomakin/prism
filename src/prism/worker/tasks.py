"""arq task functions."""

from __future__ import annotations

import logging

from prism.db.session import get_session_factory
from prism.ingest.pipeline import run_ingest

logger = logging.getLogger(__name__)


async def ingest_document(ctx: dict, job_id: int, force_reindex: bool = False) -> None:
    """Parse → chunk → embed → save for a queued Job row."""
    db = get_session_factory()()
    try:
        run_ingest(db, job_id, force_reindex=bool(force_reindex))
    finally:
        db.close()
