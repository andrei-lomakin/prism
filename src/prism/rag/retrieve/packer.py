"""ContextPacker: top-N, token budget, per-document diversity, score floor."""

from __future__ import annotations

from prism.rag.retrieve.types import RetrievedChunk


def estimate_tokens(text: str) -> int:
    # Cheap heuristic: ~4 chars/token. Enough for packing; tokenizer later.
    return max(1, len(text) // 4)


def pack_context(
    chunks: list[RetrievedChunk],
    *,
    max_chunks: int,
    max_tokens: int,
    max_per_doc: int,
    min_score: float,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    per_doc: dict[int, int] = {}
    tokens = 0
    for ch in chunks:
        if ch.score < min_score:
            continue
        if len(selected) >= max_chunks:
            break
        doc_count = per_doc.get(ch.document_id, 0)
        if doc_count >= max_per_doc:
            continue
        need = estimate_tokens(ch.text)
        if selected and tokens + need > max_tokens:
            continue
        selected.append(ch)
        per_doc[ch.document_id] = doc_count + 1
        tokens += need
    return selected
