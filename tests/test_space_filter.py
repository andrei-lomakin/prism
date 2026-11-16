from prism.rag.retrieve.spaces import apply_space_weights
from prism.rag.retrieve.types import RetrievedChunk


def _hit(i: int, space: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        document_id=i,
        document_name=f"d{i}",
        text="x",
        score=score,
        space=space,
    )


def test_space_weights_prefer_personal():
    hits = [_hit(1, "shared", 1.0), _hit(2, "personal", 0.9)]
    out = apply_space_weights(hits, {"personal": 1.0, "project": 0.85, "shared": 0.72})
    assert out[0].space == "personal"


def test_hard_filter_space():
    hits = [_hit(1, "shared"), _hit(2, "personal"), _hit(3, "shared")]
    filtered = [h for h in hits if h.space == "shared"]
    assert len(filtered) == 2
