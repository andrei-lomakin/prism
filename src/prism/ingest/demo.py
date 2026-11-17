"""Index the authored demo corpus from data/demo/corpus.

Usage: `prism ingest-demo`

Runs ingest synchronously (no Redis required) so a laptop demo works with
only Postgres up. doc_key = filename stem, stored in Document.meta.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select

from prism.config import get_settings
from prism.db.models import Document, Job
from prism.db.session import get_session_factory, init_db
from prism.ingest.pipeline import run_ingest

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = ROOT / "data" / "demo" / "corpus"


def _find_demo_doc(db, doc_key: str, filename: str) -> Document | None:
    rows = db.scalars(select(Document)).all()
    for d in rows:
        meta = d.meta or {}
        if meta.get("doc_key") == doc_key:
            return d
        if meta.get("origin") == "demo_corpus" and d.name == filename:
            return d
        if d.name == filename:
            return d
    return None


def ingest_demo_corpus(*, space: str = "shared", force_reindex: bool = False) -> dict:
    init_db()
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in CORPUS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".html", ".pdf", ".docx"}
    )
    if not files:
        raise FileNotFoundError(f"No corpus files in {CORPUS_DIR}")

    db = get_session_factory()()
    created = refreshed = failed = 0
    try:
        for path in files:
            doc_key = path.stem
            doc = _find_demo_doc(db, doc_key, path.name)
            if doc is None:
                dest = settings.upload_dir / path.name
                dest.write_bytes(path.read_bytes())
                doc = Document(
                    name=path.name,
                    source_type="file",
                    source_path=str(dest),
                    space=space,
                    status="pending",
                    meta={"doc_key": doc_key, "origin": "demo_corpus"},
                )
                db.add(doc)
                db.flush()
                created += 1
            else:
                dest = Path(doc.source_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
                doc.meta = {**(doc.meta or {}), "doc_key": doc_key, "origin": "demo_corpus"}
                db.add(doc)
                db.flush()
                refreshed += 1

            job = Job(
                document_id=doc.id,
                kind="ingest",
                status="queued",
                progress=0,
                current_step="queued",
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            try:
                run_ingest(db, job.id, force_reindex=force_reindex)
            except Exception:
                failed += 1
                logger.exception("demo ingest failed for %s", path.name)

        indexed = 0
        for d in db.scalars(select(Document)).all():
            if d.status == "indexed" and (d.meta or {}).get("origin") == "demo_corpus":
                indexed += 1

        return {
            "files": len(files),
            "new_documents": created,
            "refreshed": refreshed,
            "failed_jobs": failed,
            "indexed_demo_docs": indexed,
        }
    finally:
        db.close()
