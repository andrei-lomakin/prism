"""Ingest pipeline executed by the worker (and CLI for demo corpus)."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from prism.config import get_settings
from prism.db.models import Chunk, Document, Job
from prism.ingest.fingerprint import (
    index_fingerprint,
    index_pipeline_spec,
    stored_index_fingerprint,
)
from prism.ingest.parsers import content_hash, extract_text_from_file
from prism.rag.chunking import chunk_text
from prism.rag.embeddings import get_embedder
from prism.rag.retrieve.packer import estimate_tokens

logger = logging.getLogger(__name__)


def run_ingest(db: Session, job_id: int, *, force_reindex: bool = False) -> None:
    job = db.get(Job, job_id)
    if job is None:
        raise RuntimeError(f"job {job_id} not found")
    doc = db.get(Document, job.document_id)
    if doc is None:
        raise RuntimeError(f"document {job.document_id} not found")

    try:
        _set_job(db, job, status="running", progress=5, step="parsing")

        path = Path(doc.source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source file missing: {path}")

        text = extract_text_from_file(path)
        if not text.strip():
            raise ValueError("Document contains no extractable text")

        settings = get_settings()
        digest = content_hash(text)
        pipe_fp = index_fingerprint(settings)
        prev_hash = doc.content_hash
        prev_fp = stored_index_fingerprint(doc.meta)
        prev_chunks = doc.num_chunks or 0

        # Skip only when BOTH raw text and indexing pipeline match.
        # Otherwise ablations (C2→C5, new embedder, …) would silently keep old chunks.
        if (
            not force_reindex
            and prev_hash == digest
            and prev_fp == pipe_fp
            and prev_chunks > 0
            and doc.status == "indexed"
        ):
            has_chunks = len(doc.chunks) > 0 if doc.chunks is not None else prev_chunks > 0
            if has_chunks:
                doc.error_message = None
                db.commit()
                _set_job(db, job, status="completed", progress=100, step="skipped_unchanged")
                logger.info(
                    "Ingest skipped (content+pipeline unchanged) document_id=%s fp=%s",
                    doc.id,
                    pipe_fp[:12],
                )
                return

        if force_reindex:
            logger.info("Ingest force_reindex document_id=%s", doc.id)
        elif prev_hash == digest and prev_fp != pipe_fp:
            logger.info(
                "Ingest reindex: pipeline changed document_id=%s old_fp=%s new_fp=%s",
                doc.id,
                (prev_fp or "")[:12],
                pipe_fp[:12],
            )

        doc.status = "indexing"
        doc.error_message = None
        doc.content_hash = digest
        db.commit()

        _set_job(db, job, progress=30, step="chunking")
        # char budget ≈ 4 * token size for C1–C3 heuristic
        max_chars = settings.chunk_size * 4
        overlap_chars = settings.chunk_overlap * 4
        pieces = chunk_text(
            text,
            strategy=settings.chunking_strategy,
            max_chars=max_chars,
            overlap=overlap_chars,
        )
        if not pieces:
            raise ValueError("Chunking produced zero chunks")

        # Replace previous chunks (re-index)
        for old in list(doc.chunks):
            db.delete(old)
        db.flush()

        embedder = get_embedder(settings.embedding_model)
        if embedder.dim != settings.embedding_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: model={embedder.dim}, "
                f"DB column configured for {settings.embedding_dim}. "
                "Re-index requires a schema migration when dim changes."
            )

        _set_job(db, job, progress=50, step="embedding")

        texts = [p.text for p in pieces]
        embeddings: list[list[float]] = []
        batch = 32
        for i in range(0, len(texts), batch):
            embeddings.extend(embedder.embed_documents(texts[i : i + batch]))
            pct = 50 + int(30 * min(i + batch, len(texts)) / len(texts))
            _set_job(db, job, progress=pct, step="embedding")

        meta = dict(doc.meta or {})
        key = meta.get("doc_key") or Path(doc.name).stem or Path(doc.source_path).stem
        if isinstance(key, str) and key.startswith("demo_"):
            key = key[len("demo_") :]
        meta["doc_key"] = key
        meta["index_fingerprint"] = pipe_fp
        meta["index_pipeline"] = index_pipeline_spec(settings)

        _set_job(db, job, progress=85, step="saving")
        for i, (piece, vec) in enumerate(zip(pieces, embeddings, strict=True)):
            db.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    text=piece.text,
                    token_count=estimate_tokens(piece.text),
                    section_path=piece.section_path,
                    page=piece.page,
                    space=doc.space,
                    embedding_model=embedder.model_name,
                    embedding=vec,
                    meta={**(piece.meta or {}), "doc_key": key},
                )
            )

        doc.num_chunks = len(pieces)
        doc.status = "indexed"
        doc.error_message = None
        doc.meta = meta
        _set_job(db, job, status="completed", progress=100, step="saving")
        db.commit()
        try:
            from prism.rag.index.rebuild import rebuild_bm25

            rebuild_bm25(db)
        except Exception:
            logger.exception("BM25 rebuild after ingest failed (non-fatal)")
        logger.info("Ingest done document_id=%s chunks=%s doc_key=%s", doc.id, len(pieces), key)

    except Exception as exc:
        logger.exception("Ingest failed job_id=%s", job_id)
        db.rollback()
        job = db.get(Job, job_id)
        doc = db.get(Document, job.document_id) if job else None
        if job:
            job.status = "failed"
            job.current_step = "error"
            job.error_message = str(exc)
        if doc:
            doc.status = "error"
            doc.error_message = str(exc)
        db.commit()
        raise


def _set_job(
    db: Session,
    job: Job,
    *,
    status: str | None = None,
    progress: int | None = None,
    step: str | None = None,
) -> None:
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if step is not None:
        job.current_step = step
    db.add(job)
    db.commit()
