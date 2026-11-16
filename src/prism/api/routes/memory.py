"""Memory recall / history / soft rollback API (S13)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from prism.agents.memory import MemoryStore
from prism.api.auth import require_user
from prism.db.session import get_db

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_user)])


class RememberBody(BaseModel):
    layer: str = Field(pattern="^(profile|journal|project)$")
    key: str
    content: str
    meta: dict[str, Any] | None = None


@router.get("/recall")
def recall(
    q: str = "",
    layer: str | None = None,
    limit: int = 10,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    mem = MemoryStore(db)
    layers = [layer] if layer else None
    hits = mem.recall(q, layers=layers, limit=min(50, max(1, limit)))
    return {
        "items": [
            {"id": h.id, "layer": h.layer, "key": h.key, "content": h.content, "meta": h.meta}
            for h in hits
        ]
    }


@router.post("/remember")
def remember(body: RememberBody, db: Session = Depends(get_db)) -> dict[str, Any]:
    mem = MemoryStore(db)
    entry_id = mem.remember(body.layer, body.key, body.content, meta=body.meta)
    db.commit()
    return {"id": entry_id, "ok": True}


@router.get("/history")
def history(layer: str, key: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    mem = MemoryStore(db)
    hits = mem.history(layer, key)
    return {
        "items": [
            {"id": h.id, "layer": h.layer, "key": h.key, "content": h.content, "meta": h.meta}
            for h in hits
        ]
    }


@router.post("/rollback/{entry_id}")
def rollback(entry_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    mem = MemoryStore(db)
    out = mem.rollback(entry_id)
    if not out.get("ok"):
        raise HTTPException(404, out.get("error", "not_found"))
    db.commit()
    return out
