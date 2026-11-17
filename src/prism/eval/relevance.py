"""Map golden labels ↔ retrieved hits.

Retrieval Score@k (Recall/MRR/nDCG/…): **doc_key** only.
Context precision/recall: section-aware units ``(doc_key, section?)``.
"""

from __future__ import annotations

from prism.eval.schema import GoldenQuestion
from prism.rag.retrieve.types import RetrievedChunk


def gold_doc_keys(q: GoldenQuestion) -> set[str]:
    return {d.doc_key for d in q.relevant_docs}


def gold_evidence_units(q: GoldenQuestion) -> list[tuple[str, str | None]]:
    """Ordered gold evidence for context recall (doc_key + optional section)."""
    return [(d.doc_key, d.section) for d in q.relevant_docs]


def retrieved_doc_keys(hits: list[RetrievedChunk]) -> list[str]:
    """Preserve rank order; first occurrence of each doc_key wins for MRR/nDCG at doc level."""
    seen: set[str] = set()
    ordered: list[str] = []
    for h in hits:
        key = h.doc_key or f"doc:{h.document_id}"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def retrieved_context_units(hits: list[RetrievedChunk]) -> list[tuple[str, str | None]]:
    """Per-chunk (doc_key, section_path) in rank order — for context metrics."""
    out: list[tuple[str, str | None]] = []
    for h in hits:
        key = h.doc_key or f"doc:{h.document_id}"
        out.append((key, h.section_path))
    return out


def retrieved_chunk_ids(hits: list[RetrievedChunk]) -> list[str]:
    return [str(h.chunk_id) for h in hits]
