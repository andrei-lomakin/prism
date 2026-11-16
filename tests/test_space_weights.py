from prism.rag.retrieve.spaces import apply_space_weights
from prism.rag.retrieve.types import RetrievedChunk


def _ch(cid: int, space: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=cid,
        document_name=f"{space}.md",
        text="x",
        score=score,
        space=space,
        doc_key=space,
    )


def test_personal_outranks_shared_on_equal_raw_score():
    chunks = [_ch(1, "shared", 1.0), _ch(2, "personal", 1.0)]
    out = apply_space_weights(chunks)
    assert out[0].space == "personal"
    assert out[0].score > out[1].score
