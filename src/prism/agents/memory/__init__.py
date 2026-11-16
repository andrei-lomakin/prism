"""Layered memory store: profile / journal / project with soft supersede."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from prism.db.models import MemoryEntry


@dataclass
class MemoryHit:
    id: int
    layer: str
    key: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """In-memory or DB-backed. Soft supersede — never hard-delete."""

    def __init__(self, db: Session | None = None) -> None:
        self.db = db
        self._local: list[dict[str, Any]] = []
        self._seq = 0

    def remember(
        self,
        layer: str,
        key: str,
        content: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> int:
        meta = meta or {}
        if self.db is not None:
            prev = (
                self.db.query(MemoryEntry)
                .filter(
                    MemoryEntry.layer == layer,
                    MemoryEntry.key == key,
                    MemoryEntry.active.is_(True),
                )
                .order_by(MemoryEntry.id.desc())
                .first()
            )
            if prev is not None:
                prev.active = False
            entry = MemoryEntry(
                layer=layer,
                key=key,
                content=content,
                active=True,
                supersedes_id=prev.id if prev else None,
                meta=meta,
            )
            self.db.add(entry)
            self.db.flush()
            return int(entry.id)

        # Local fallback for unit tests
        for row in self._local:
            if row["layer"] == layer and row["key"] == key and row["active"]:
                row["active"] = False
                supersedes = row["id"]
                break
        else:
            supersedes = None
        self._seq += 1
        self._local.append(
            {
                "id": self._seq,
                "layer": layer,
                "key": key,
                "content": content,
                "active": True,
                "supersedes_id": supersedes,
                "meta": meta,
            }
        )
        return self._seq

    def recall(
        self,
        query: str,
        layers: list[str] | None = None,
        *,
        limit: int = 10,
    ) -> list[MemoryHit]:
        q = (query or "").strip().lower()
        hits: list[MemoryHit] = []

        if self.db is not None:
            rows = (
                self.db.query(MemoryEntry)
                .filter(MemoryEntry.active.is_(True))
                .order_by(MemoryEntry.id.desc())
                .all()
            )
            for row in rows:
                if layers and row.layer not in layers:
                    continue
                if q and q not in row.key.lower() and q not in row.content.lower():
                    continue
                hits.append(
                    MemoryHit(
                        id=row.id,
                        layer=row.layer,
                        key=row.key,
                        content=row.content,
                        meta=dict(row.meta or {}),
                    )
                )
                if len(hits) >= limit:
                    break
            return hits

        for row in reversed(self._local):
            if not row["active"]:
                continue
            if layers and row["layer"] not in layers:
                continue
            if q and q not in row["key"].lower() and q not in row["content"].lower():
                continue
            hits.append(
                MemoryHit(
                    id=row["id"],
                    layer=row["layer"],
                    key=row["key"],
                    content=row["content"],
                    meta=dict(row["meta"] or {}),
                )
            )
            if len(hits) >= limit:
                break
        return hits
