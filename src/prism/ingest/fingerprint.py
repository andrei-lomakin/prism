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


class IndexFingerprintError(RuntimeError):
    """Eval corpus is missing or was built with a different index pipeline."""


def require_index_fingerprint(documents: list[Any], settings: Settings) -> str:
    """Stop eval when indexed documents were not built with ``settings``.

    Compares each indexed document's stored fingerprint to the fingerprint of the
    experiment config. Does not reindex.
    """
    expected = index_fingerprint(settings)
    indexed = [doc for doc in documents if getattr(doc, "status", None) == "indexed"]
    spec = (
        f"chunking={settings.chunking_strategy} embedding={settings.embedding_model} "
        f"fingerprint={expected[:12]}"
    )
    if not indexed:
        raise IndexFingerprintError(
            "Eval aborted: no indexed documents. Reindex the corpus with this "
            f"experiment config ({spec})."
        )
    bad: list[str] = []
    for doc in indexed:
        got = stored_index_fingerprint(getattr(doc, "meta", None))
        if got != expected:
            name = getattr(doc, "name", None) or getattr(doc, "id", "?")
            bad.append(f"{name} fp={(got or 'missing')[:12]}")
    if bad:
        preview = "; ".join(bad[:8])
        extra = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        raise IndexFingerprintError(
            "Eval aborted: stored index does not match the experiment config "
            f"(expected {spec}). Reindex before eval. Mismatched: {preview}{extra}"
        )
    return expected
