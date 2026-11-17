"""retrieve(pack=False) must not call pack_context."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from prism.config import Settings
from prism.rag.retrieve.service import retrieve
from prism.rag.retrieve.types import RetrievedChunk


def _hit(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        document_id=i,
        document_name=f"doc{i}",
        text=f"text {i}",
        score=1.0 - i * 0.01,
        doc_key=f"key{i}",
    )


def test_retrieve_pack_false_skips_packer():
    hits = [_hit(i) for i in range(12)]
    settings = Settings(
        retrieval_mode="dense",
        top_k=5,
        rewrite_strategy="R0",
        rerank_enabled=False,
        space_weights_enabled=False,
        pack_max_chunks=2,
        pack_max_per_doc=1,
    )
    with (
        patch("prism.rag.retrieve.service._dense_search", return_value=hits),
        patch("prism.rag.retrieve.service.pack_context") as pack_mock,
        patch("prism.rag.retrieve.service.rewrite_query", return_value={"semantic_query": "q"}),
    ):
        out, mode = retrieve(MagicMock(), "q", top_k=5, settings=settings, pack=False)
    pack_mock.assert_not_called()
    assert len(out) == 5
    assert "nopack" in mode


def test_retrieve_pack_true_calls_packer():
    hits = [_hit(i) for i in range(8)]
    packed = hits[:2]
    settings = Settings(
        retrieval_mode="dense",
        top_k=5,
        rewrite_strategy="R0",
        rerank_enabled=False,
        space_weights_enabled=False,
    )
    with (
        patch("prism.rag.retrieve.service._dense_search", return_value=hits),
        patch("prism.rag.retrieve.service.pack_context", return_value=packed) as pack_mock,
        patch("prism.rag.retrieve.service.rewrite_query", return_value={"semantic_query": "q"}),
    ):
        out, mode = retrieve(MagicMock(), "q", top_k=5, settings=settings, pack=True)
    pack_mock.assert_called_once()
    assert out == packed
    assert "nopack" not in mode
