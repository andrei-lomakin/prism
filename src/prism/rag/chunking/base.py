"""Chunking strategies C1–C7.

C1 fixed word-window, word-safe overlap
C2 paragraph packing
C3 sentence packing
C4 recursive: paragraph → sentence → word until size fits
C5 structure-aware: Markdown/HTML headings + PDF page markers → section_path
C6 semantic: split where adjacent-sentence Jaccard similarity drops
C7 parent-child: small children for search; parent text in meta for generation
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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
_PAGE_MARK = re.compile(r"<!--\s*page:(\d+)\s*-->", re.I)
_HTML_HEADING = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.I | re.S)


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
        parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        return [ChunkPiece(text=c) for c in _pack_parts(parts, max_chars, overlap)]
    if strategy == "C4":
        return _chunk_recursive(text, max_chars, overlap)
    if strategy == "C5":
        return _chunk_structure_aware(text, max_chars, overlap)
    if strategy == "C6":
        return _chunk_semantic(text, max_chars, overlap)
    if strategy == "C7":
        return _chunk_parent_child(text, max_chars, overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _chunk_recursive(text: str, max_size: int, overlap: int) -> list[ChunkPiece]:
    """Paragraph → sentence → word until each piece fits max_size."""

    def split_unit(unit: str) -> list[str]:
        if len(unit) <= max_size:
            return [unit]
        paras = [p.strip() for p in re.split(r"\n\s*\n", unit) if p.strip()]
        if len(paras) > 1:
            out: list[str] = []
            for p in paras:
                out.extend(split_unit(p))
            return out
        sents = [s.strip() for s in _SENTENCE_SPLIT.split(unit) if s.strip()]
        if len(sents) > 1:
            out = []
            for s in sents:
                out.extend(split_unit(s))
            return out
        return _chunk_by_words(unit, max_size, overlap)

    parts = split_unit(text)
    packed = _pack_parts(parts, max_size, overlap)
    return [ChunkPiece(text=c, meta={"strategy": "C4"}) for c in packed]


def _chunk_structure_aware(text: str, max_size: int, overlap: int) -> list[ChunkPiece]:
    """Split on MD/HTML headings and PDF page markers; keep section_path/page."""
    # Normalize HTML headings to markdown-like lines for one code path
    normalized = _HTML_HEADING.sub(
        lambda m: f"\n{'#' * int(m.group(1))} {re.sub('<[^>]+>', '', m.group(2)).strip()}\n",
        text,
    )

    sections: list[tuple[str | None, int | None, str]] = []
    current_path: list[str] = []
    current_page: int | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            path = " / ".join(current_path) if current_path else None
            sections.append((path, current_page, body))
        buf.clear()

    for line in normalized.splitlines():
        page_m = _PAGE_MARK.search(line)
        if page_m:
            flush()
            current_page = int(page_m.group(1))
            continue
        hm = re.match(r"^(#{1,6})\s+(.+)$", line)
        if hm:
            flush()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            current_path = current_path[: level - 1] + [title]
            continue
        buf.append(line)
    flush()

    if not sections:
        return [
            ChunkPiece(text=c, meta={"strategy": "C5"})
            for c in _pack_parts(
                [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()],
                max_size,
                overlap,
            )
        ]

    out: list[ChunkPiece] = []
    for path, page, body in sections:
        for piece in _pack_parts(
            [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()] or [body],
            max_size,
            overlap,
        ):
            out.append(
                ChunkPiece(
                    text=piece,
                    section_path=path,
                    page=page,
                    meta={"strategy": "C5"},
                )
            )
    return out


def _chunk_semantic(text: str, max_size: int, overlap: int, *, threshold: float = 0.15) -> list[ChunkPiece]:
    """Boundary where Jaccard similarity between adjacent sentences drops."""
    sents = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1:
        return [ChunkPiece(text=text, meta={"strategy": "C6"})]

    def toks(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", s)}

    groups: list[list[str]] = [[sents[0]]]
    for prev, cur in zip(sents, sents[1:]):
        a, b = toks(prev), toks(cur)
        sim = (len(a & b) / len(a | b)) if (a or b) else 1.0
        if sim < threshold:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    pieces: list[ChunkPiece] = []
    for g in groups:
        block = " ".join(g)
        parts = (
            _chunk_by_words(block, max_size, overlap)
            if len(block) > max_size
            else [block]
        )
        for c in parts:
            pieces.append(
                ChunkPiece(text=c, meta={"strategy": "C6", "semantic_threshold": threshold})
            )
    return pieces


def _chunk_parent_child(text: str, max_size: int, overlap: int) -> list[ChunkPiece]:
    """Small children (~max_size/2) with parent paragraph text in meta."""
    child_size = max(64, max_size // 2)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]
    out: list[ChunkPiece] = []
    for pi, para in enumerate(paras):
        children = _chunk_by_words(para, child_size, max(0, overlap // 2)) or [para]
        for ci, child in enumerate(children):
            out.append(
                ChunkPiece(
                    text=child,
                    meta={
                        "strategy": "C7",
                        "parent_text": para[: max_size * 2],
                        "parent_index": pi,
                        "child_index": ci,
                    },
                )
            )
    return out


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
