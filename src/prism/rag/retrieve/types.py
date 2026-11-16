"""Shared retrieval DTOs (kept separate to avoid circular imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_name: str
    text: str
    score: float
    section_path: str | None = None
    page: int | None = None
    space: str = "personal"
    doc_key: str | None = None
    meta: dict | None = None
