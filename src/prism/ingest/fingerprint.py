"""Index pipeline fingerprint — skip-unchanged must include config, not only text."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from prism.config import Settings

# Bump when preprocessing / parser semantics change without touching Settings fields.
PREPROCESSING_VERSION = "1"


def index_pipeline_spec(settings: Settings) -> dict[str, Any]:
    """Canonical knobs that change chunk identity or vector geometry."""
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "chunking_strategy": settings.chunking_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
    }


def index_fingerprint(settings: Settings) -> str:
    """Stable SHA-256 over pipeline spec (not document text)."""
    payload = json.dumps(index_pipeline_spec(settings), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stored_index_fingerprint(doc_meta: dict[str, Any] | None) -> str | None:
    meta = doc_meta or {}
    fp = meta.get("index_fingerprint")
    return str(fp) if fp else None
