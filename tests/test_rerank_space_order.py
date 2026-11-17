"""Integration-style tests: space weights before rerank; final CE order wins."""

from __future__ import annotations

from prism.rag.retrieve.service import apply_cross_encoder_rerank
from prism.rag.retrieve.spaces import apply_space_weights
from prism.rag.retrieve.types import RetrievedChunk


def _ch(cid: int, space: str, score: float, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=cid,
        document_name=f"{space}.md",
        text=text,
        score=score,
        space=space,
        doc_key=space,
    )


def _fake_rerank(query: str, texts: list[str], top_n: int, model_name: str = ""):
    """Prefer texts containing 'WIN'; ignore original retrieval order."""
    scored = []
    for i, t in enumerate(texts):
        scored.append((i, 10.0 if "WIN" in t else 0.1))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def test_pipeline_space_then_rerank_preserves_ce_order():
    """Recommended pipeline: space_weights → cross-encoder (final)."""
    # Retrieval prefers shared (high raw score). Space weight should boost personal.
    shared = _ch(1, "shared", 0.99, text="shared LOSE")
    personal = _ch(2, "personal", 0.20, text="personal WIN")

    after_space = apply_space_weights(
        [shared, personal],
        {"personal": 1.0, "project": 0.85, "shared": 0.72},
    )
    # personal: 0.20*1.0=0.20; shared: 0.99*0.72=0.7128 → shared still first by space
    assert after_space[0].chunk_id == 1
    assert after_space[0].meta["score_retrieval"] == 0.99
    assert after_space[0].meta["score_space_weighted"] == after_space[0].score

    final = apply_cross_encoder_rerank(
        after_space,
        "q",
        top_n=2,
        model_name="mock",
        rerank_fn=_fake_rerank,
    )
    # Cross-encoder must decide the last order (WIN first), not old retrieval.
    assert [c.chunk_id for c in final] == [2, 1]
    assert final[0].meta["rerank_score"] == 10.0
    assert final[0].score == 10.0
    assert "score_pre_rerank" in final[0].meta


def test_rerank_without_space_still_sets_ce_score():
    a = _ch(1, "shared", 0.1, text="LOSE")
    b = _ch(2, "shared", 0.9, text="WIN")
    # Wrong retrieval order on purpose
    final = apply_cross_encoder_rerank(
        [a, b],
        "q",
        top_n=2,
        model_name="mock",
        rerank_fn=_fake_rerank,
    )
    assert [c.chunk_id for c in final] == [2, 1]


def test_bug_regression_space_after_rerank_would_undo_order():
    """Document the old bug: space_weights AFTER rerank using pre-CE scores undoes CE.

    We keep this as a negative control — what NOT to do.
    """
    # Pretend CE ordered WIN first but left old retrieval scores on .score (the bug).
    win = _ch(2, "shared", score=0.1, text="WIN")  # CE winner, old retrieval low
    lose = _ch(1, "shared", score=0.9, text="LOSE")  # CE loser, old retrieval high
    wrongly = apply_space_weights(
        [win, lose],
        {"personal": 1.0, "project": 1.0, "shared": 1.0},
    )
    # Uniform weights + old scores → LOSE jumps back to front (bug symptom).
    assert [c.chunk_id for c in wrongly] == [1, 2]


def test_space_truncation_can_drop_chunk_before_ce():
    """Space-before-rerank only matters if the CE pool is truncated after weighting."""
    chunks = [
        _ch(1, "shared", 1.0, text="shared A"),
        _ch(2, "shared", 0.9, text="shared B"),
        _ch(3, "personal", 0.5, text="personal WIN"),
    ]
    weighted = apply_space_weights(
        chunks,
        {"personal": 1.0, "project": 0.85, "shared": 0.4},
    )
    # personal 0.5 > shared 1.0*0.4=0.4 and 0.9*0.4=0.36
    assert weighted[0].chunk_id == 3
    pool = weighted[:2]
    assert {c.chunk_id for c in pool} == {3, 1}
    final = apply_cross_encoder_rerank(
        pool,
        "q",
        top_n=2,
        model_name="mock",
        rerank_fn=_fake_rerank,
    )
    assert final[0].chunk_id == 3
    # chunk 2 never entered CE pool — space truncation worked
    assert all(c.chunk_id != 2 for c in final)
