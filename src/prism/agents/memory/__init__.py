"""Layered memory store: profile / journal / project with supersede + rollback."""

from __future__ import annotations


class MemoryStore:
    def remember(self, layer: str, key: str, content: str) -> int:
        raise NotImplementedError("MemoryStore lands in stage 4")

    def recall(self, query: str, layers: list[str] | None = None) -> list[dict]:
        raise NotImplementedError("MemoryStore lands in stage 4")
