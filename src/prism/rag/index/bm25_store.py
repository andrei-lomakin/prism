"""BM25 index per collection (bm25s). Stub for stage 3."""

from __future__ import annotations

from pathlib import Path


class BM25Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        raise NotImplementedError("BM25 index lands in stage 3")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        raise NotImplementedError("BM25 search lands in stage 3")
