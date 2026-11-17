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


def test_uniform_space_weights_preserve_rerank_order():
    """Regression: after rerank, score must be the rerank score.

    If space weights keep the old retrieval score, re-sorting undoes rerank
    even when every space weight is 1.0.
    """
    # Simulated post-rerank list: order A then B, scores from cross-encoder.
    # Old retrieval scores (if wrongly kept) would prefer B.
    a = _ch(1, "shared", score=0.9)  # rerank winner
    b = _ch(2, "shared", score=0.1)  # rerank loser
    out = apply_space_weights([a, b], {"personal": 1.0, "project": 1.0, "shared": 1.0})
    assert [c.chunk_id for c in out] == [1, 2]


def test_space_weight_still_can_override_close_rerank_scores():
    personal = _ch(1, "personal", score=0.80)
    shared = _ch(2, "shared", score=0.85)
    out = apply_space_weights(
        [shared, personal],
        {"personal": 1.0, "project": 0.85, "shared": 0.72},
    )
    # 0.80*1.0=0.80 > 0.85*0.72=0.612
    assert out[0].chunk_id == 1

