"""Chunking registry.

C1 fixed word-window with overlap (overlap never splits mid-word)
C2 paragraphs packed to size budget
C3 sentences packed to size budget
C4–C7: stubs — raise NotImplementedError until stage 3
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChunkPiece:
    text: str
    section_path: str | None = None
    page: int | None = None
    meta: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    *,
    strategy: str,
    max_chars: int,
    overlap: int,
) -> list[ChunkPiece]:
    text = text.strip()
    if not text:
        return []

    if strategy in ("C1", "by_tokens"):
        return [ChunkPiece(text=c) for c in _chunk_by_words(text, max_chars, overlap)]
    if strategy in ("C2", "by_paragraph"):
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [ChunkPiece(text=c) for c in _pack_parts(parts, max_chars, overlap)]
    if strategy in ("C3", "by_sentence"):
        parts = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        return [ChunkPiece(text=c) for c in _pack_parts(parts, max_chars, overlap)]
    if strategy in ("C4", "C5", "C6", "C7"):
        raise NotImplementedError(f"Chunking strategy {strategy} lands in stage 3")
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _pack_parts(parts: list[str], max_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for part in parts:
        if len(part) > max_size:
            flush()
            chunks.extend(_chunk_by_words(part, max_size, overlap))
            continue
        candidate = f"{buf}\n\n{part}".strip() if buf else part
        if len(candidate) <= max_size:
            buf = candidate
        else:
            flush()
            buf = part
    flush()

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    # Word-safe overlap: take whole words from the previous chunk tail.
    out: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        tail: list[str] = []
        size = 0
        for w in reversed(prev_words):
            add = len(w) + (1 if tail else 0)
            if size + add > overlap and tail:
                break
            tail.insert(0, w)
            size += add
        prefix = " ".join(tail)
        out.append(f"{prefix} {chunks[i]}".strip() if prefix else chunks[i])
    return out


def _chunk_by_words(text: str, max_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        size = 0
        while end < len(words):
            wlen = len(words[end]) + (1 if end > start else 0)
            if size + wlen > max_size and end > start:
                break
            size += wlen
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # Step back by ~overlap characters on word boundaries.
        back = 0
        i = end - 1
        while i > start and back < overlap:
            back += len(words[i]) + 1
            i -= 1
        next_start = max(i + 1, start + 1)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
