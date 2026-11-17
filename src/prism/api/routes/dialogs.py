"""Persisted chat dialogs: list / get / upsert / delete."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from prism.api.auth import require_user
from prism.db.models import ChatDialog
from prism.db.session import get_db

router = APIRouter(prefix="/api/dialogs", tags=["dialogs"], dependencies=[Depends(require_user)])


class DialogMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20000)
    citations: list[dict[str, Any]] | None = None


class DialogUpsert(BaseModel):
    id: int | None = None
    title: str | None = Field(default=None, max_length=256)
    messages: list[DialogMessage] = Field(default_factory=list, max_length=64)
    meta: dict[str, Any] = Field(default_factory=dict)


def _title_from_messages(messages: list[DialogMessage], fallback: str = "Диалог") -> str:
    for m in messages:
        if m.role == "user" and m.content.strip():
            t = " ".join(m.content.strip().split())
            return (t[:72] + "…") if len(t) > 72 else t
    return fallback


def _serialize(row: ChatDialog) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "messages": row.messages or [],
        "meta": row.meta or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "n_messages": len(row.messages or []),
    }


@router.get("")
def list_dialogs(limit: int = 50, db: Session = Depends(get_db)) -> dict[str, Any]:
    lim = max(1, min(100, limit))
    rows = db.scalars(select(ChatDialog).order_by(ChatDialog.updated_at.desc()).limit(lim)).all()
    return {
        "items": [
            {
                "id": r.id,
                "title": r.title,
                "n_messages": len(r.messages or []),
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]
    }


@router.get("/{dialog_id}")
def get_dialog(dialog_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.get(ChatDialog, dialog_id)
    if row is None:
        raise HTTPException(404, "dialog not found")
    return _serialize(row)


@router.post("")
def upsert_dialog(payload: DialogUpsert, db: Session = Depends(get_db)) -> dict[str, Any]:
    msgs = [m.model_dump(exclude_none=True) for m in payload.messages]
    # Cap citation text to keep JSONB lean
    for m in msgs:
        cites = m.get("citations") or []
        slim = []
        for c in cites[:12]:
            if not isinstance(c, dict):
                continue
            slim.append(
                {
                    "n": c.get("n"),
                    "title": c.get("title") or c.get("document_name"),
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "page": c.get("page"),
                    "score": c.get("score"),
                    "section_path": c.get("section_path"),
                    "url": c.get("url"),
                    "kind": c.get("kind"),
                    "text": (str(c.get("text") or ""))[:500] or None,
                }
            )
        if slim:
            m["citations"] = slim
        elif "citations" in m:
            del m["citations"]

    title = (payload.title or "").strip() or _title_from_messages(payload.messages)

    if payload.id is not None:
        row = db.get(ChatDialog, payload.id)
        if row is None:
            raise HTTPException(404, "dialog not found")
        row.title = title
        row.messages = msgs
        row.meta = payload.meta or {}
    else:
        row = ChatDialog(title=title, messages=msgs, meta=payload.meta or {})
        db.add(row)

    db.commit()
    db.refresh(row)
    return _serialize(row)


@router.delete("/{dialog_id}")
def delete_dialog(dialog_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.get(ChatDialog, dialog_id)
    if row is None:
        raise HTTPException(404, "dialog not found")
    db.delete(row)
    db.commit()
    return {"ok": True, "id": dialog_id}
