"""Map golden labels ↔ retrieved hits.

Relevance is at **doc_key** level (TZ): a retrieved chunk is a hit if its
document's doc_key is in the gold relevant set. Section is soft signal for
diagnostics, not required for Score@k in Stage 2.
"""

from __future__ import annotations

from prism.eval.schema import GoldenQuestion
from prism.rag.retrieve.types import RetrievedChunk


def gold_doc_keys(q: GoldenQuestion) -> set[str]:
    return {d.doc_key for d in q.relevant_docs}


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


def retrieved_chunk_ids(hits: list[RetrievedChunk]) -> list[str]:
    return [str(h.chunk_id) for h in hits]
