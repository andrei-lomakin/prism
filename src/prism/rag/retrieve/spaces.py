"""S1: space hierarchy weights — personal → project → shared."""

from __future__ import annotations

from prism.config import Settings
from prism.rag.retrieve.types import RetrievedChunk

DEFAULT_WEIGHTS = {
    "personal": 1.0,
    "project": 0.85,
    "shared": 0.72,
}


def space_weights_from_settings(settings: Settings) -> dict[str, float]:
    return {
        "personal": float(settings.space_weight_personal),
        "project": float(settings.space_weight_project),
        "shared": float(settings.space_weight_shared),
    }


def apply_space_weights(
    chunks: list[RetrievedChunk],
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Multiply retrieval score by space weight and re-sort desc.

    Writes explicit score fields into meta:
      score_retrieval — input score
      space_weight — factor for this space
      score_space_weighted — product (also copied to chunk.score)
    """
    w = weights or DEFAULT_WEIGHTS
    out: list[RetrievedChunk] = []
    for ch in chunks:
        factor = float(w.get(ch.space, 1.0))
        raw = float(ch.score)
        weighted = raw * factor
        meta = {
            **(ch.meta or {}),
            "score_retrieval": raw,
            "score_raw": raw,  # alias kept for older traces
            "space_weight": factor,
            "score_space_weighted": weighted,
        }
        out.append(
            RetrievedChunk(
                chunk_id=ch.chunk_id,
                document_id=ch.document_id,
                document_name=ch.document_name,
                text=ch.text,
                score=weighted,
                section_path=ch.section_path,
                page=ch.page,
                space=ch.space,
                doc_key=ch.doc_key,
                meta=meta,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out
