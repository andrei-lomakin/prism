"""Ingest uses the job settings snapshot and does not drop a live index on failure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prism.db.models import Document, Job
from prism.ingest.pipeline import run_ingest
from prism.rag.embeddings.base import HashEmbedder


def _session(job, doc):
    class Session:
        def __init__(self) -> None:
            self.pending_deletes: list = []
            self.added: list = []
            self._committed = self._capture()

        def _capture(self) -> dict:
            return {
                "status": doc.status,
                "content_hash": doc.content_hash,
                "meta": dict(doc.meta or {}),
                "num_chunks": doc.num_chunks,
                "error_message": doc.error_message,
                "chunks": list(doc.chunks),
                "job_status": job.status,
                "job_step": job.current_step,
                "job_progress": job.progress,
                "job_error": job.error_message,
            }

        def _restore(self, snap: dict) -> None:
            doc.status = snap["status"]
            doc.content_hash = snap["content_hash"]
            doc.meta = dict(snap["meta"])
            doc.num_chunks = snap["num_chunks"]
            doc.error_message = snap["error_message"]
            doc.chunks = list(snap["chunks"])
            job.status = snap["job_status"]
            job.current_step = snap["job_step"]
            job.progress = snap["job_progress"]
            job.error_message = snap["job_error"]

        def get(self, model, ident):
            if model is Job:
                return job
            if model is Document:
                return doc
            raise AssertionError(model)

        def delete(self, obj) -> None:
            self.pending_deletes.append(obj)

        def add(self, obj) -> None:
            self.added.append(obj)

        def flush(self) -> None:
            return None

        def commit(self) -> None:
            for obj in self.pending_deletes:
                if obj in doc.chunks:
                    doc.chunks.remove(obj)
            self.pending_deletes.clear()
            for obj in self.added:
                if hasattr(obj, "chunk_index") and hasattr(obj, "embedding_model"):
                    doc.chunks.append(obj)
            self.added.clear()
            self._committed = self._capture()

        def rollback(self) -> None:
            self.pending_deletes.clear()
            self.added.clear()
            self._restore(self._committed)

    return Session()


def _job_and_doc(path, *, strategy: str = "C5"):
    old = SimpleNamespace(text="OLD-CHUNK", chunk_index=0)
    doc = SimpleNamespace(
        id=1,
        name="doc.md",
        source_path=str(path),
        space="personal",
        status="indexed",
        content_hash="old-hash",
        num_chunks=1,
        error_message=None,
        meta={"doc_key": "doc", "index_fingerprint": "stale"},
        chunks=[old],
    )
    job = SimpleNamespace(
        id=7,
        document_id=1,
        status="queued",
        progress=0,
        current_step="queued",
        error_message=None,
        payload={
            "ingest_settings": {
                "embedding_model": "ui-embedder-b",
                "embedding_dim": 384,
                "chunking_strategy": strategy,
                "chunk_size": 256,
                "chunk_overlap": 32,
            }
        },
    )
    return job, doc, old


def test_failed_reindex_keeps_previous_chunks(monkeypatch, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody of the document.\n", encoding="utf-8")
    job, doc, old = _job_and_doc(path)
    db = _session(job, doc)
    seen: list[str] = []

    class Boom(HashEmbedder):
        def embed_documents(self, texts):
            assert db.pending_deletes == []
            assert old in doc.chunks
            raise RuntimeError("embed failed")

    def fake_embedder(name: str) -> Boom:
        seen.append(name)
        return Boom(dim=384, model_name=name)

    monkeypatch.setattr("prism.ingest.pipeline.get_embedder", fake_embedder)
    with pytest.raises(RuntimeError, match="embed failed"):
        run_ingest(db, job.id)
    assert seen == ["ui-embedder-b"]
    assert doc.chunks == [old]
    assert doc.status == "indexed"
    assert doc.content_hash == "old-hash"
    assert "previous index kept" in (doc.error_message or "")
    assert job.status == "failed"


def test_successful_reindex_uses_snapshot_pipeline(monkeypatch, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody of the document.\n\n## Section\n\nMore.\n", encoding="utf-8")
    job, doc, old = _job_and_doc(path, strategy="C5")
    db = _session(job, doc)

    def fake_embedder(name: str) -> HashEmbedder:
        return HashEmbedder(dim=384, model_name=name)

    monkeypatch.setattr("prism.ingest.pipeline.get_embedder", fake_embedder)
    monkeypatch.setattr("prism.rag.index.rebuild.rebuild_bm25", lambda _db: 0)
    run_ingest(db, job.id)
    assert old not in doc.chunks
    assert doc.chunks
    assert {c.embedding_model for c in doc.chunks} == {"ui-embedder-b"}
    assert any((c.meta or {}).get("strategy") == "C5" for c in doc.chunks)
    assert any(c.section_path for c in doc.chunks)
    spec = doc.meta["index_pipeline"]
    assert spec["chunking_strategy"] == "C5"
    assert spec["embedding_model"] == "ui-embedder-b"
    assert doc.meta["index_fingerprint"]
    assert doc.status == "indexed"
    assert job.status == "completed"
