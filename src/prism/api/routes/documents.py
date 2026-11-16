"""Document upload / URL ingest / listing."""

from __future__ import annotations

import uuid
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from prism.api.schemas import (
    DocumentListResponse,
    DocumentOut,
    JobOut,
    UploadResponse,
    UrlIngestRequest,
)
from prism.config import get_settings
from prism.db.models import Document, Job
from prism.db.session import get_db
from prism.ingest.parsers import (
    ALLOWED_EXTENSIONS,
    download_url_to_file,
    safe_filename,
    suggest_name_from_url,
)

router = APIRouter(prefix="/api", tags=["documents"])


def _doc_out(d: Document) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        name=d.name,
        source_type=d.source_type,
        source_path=d.source_path,
        source_url=d.source_url,
        space=d.space,
        status=d.status,
        num_chunks=d.num_chunks,
        error_message=d.error_message,
        created_at=d.created_at.isoformat() if d.created_at else "",
        updated_at=d.updated_at.isoformat() if d.updated_at else "",
    )


async def _enqueue(job_id: int) -> None:
    settings = get_settings()
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        await redis.enqueue_job("ingest_document", job_id)
    finally:
        await redis.close()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    space: str = "personal",
    db: Session = Depends(get_db),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(400, "Empty filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported type {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored = f"{safe_filename(file.filename)}_{uuid.uuid4().hex[:8]}{suffix}"
    dest = settings.upload_dir / stored
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    dest.write_bytes(content)

    doc = Document(
        name=file.filename,
        source_type="file",
        source_path=str(dest),
        space=space or settings.space_default,
        status="pending",
    )
    db.add(doc)
    db.flush()
    job = Job(document_id=doc.id, kind="ingest", status="queued", progress=0, current_step="queued")
    db.add(job)
    db.commit()
    db.refresh(doc)
    db.refresh(job)
    await _enqueue(job.id)
    return UploadResponse(document_id=doc.id, job_id=job.id)


@router.post("/documents/from-url", response_model=UploadResponse)
async def ingest_url(payload: UrlIngestRequest, db: Session = Depends(get_db)) -> UploadResponse:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    url = str(payload.url)
    dest = settings.upload_dir / f"url_{uuid.uuid4().hex[:10]}"
    try:
        saved = download_url_to_file(url, dest)
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch URL: {exc}") from exc

    doc = Document(
        name=suggest_name_from_url(url),
        source_type="url",
        source_path=str(saved),
        source_url=url,
        space=payload.space,
        status="pending",
    )
    db.add(doc)
    db.flush()
    job = Job(document_id=doc.id, kind="ingest", status="queued", progress=0, current_step="queued")
    db.add(job)
    db.commit()
    db.refresh(doc)
    db.refresh(job)
    await _enqueue(job.id)
    return UploadResponse(document_id=doc.id, job_id=job.id)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    page: int = 1, page_size: int = 10, db: Session = Depends(get_db)
) -> DocumentListResponse:
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    total = db.scalar(select(func.count()).select_from(Document)) or 0
    pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = min(page, pages)
    rows = db.scalars(
        select(Document)
        .order_by(Document.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return DocumentListResponse(
        items=[_doc_out(d) for d in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentOut:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return _doc_out(doc)


@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)) -> dict:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    if doc.status in ("pending", "indexing"):
        raise HTTPException(409, "Wait for indexing to finish")
    path = Path(doc.source_path)
    n_chunks = len(doc.chunks)
    db.delete(doc)
    db.commit()
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
    return {"document_id": document_id, "deleted_chunks": n_chunks}


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobOut(
        id=job.id,
        document_id=job.document_id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        error_message=job.error_message,
    )
