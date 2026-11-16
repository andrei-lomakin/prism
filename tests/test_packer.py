from prism.rag.retrieve.packer import pack_context
from prism.rag.retrieve.types import RetrievedChunk


def _hit(doc: int, score: float, text: str = "x" * 40) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=doc * 10,
        document_id=doc,
        document_name=f"d{doc}",
        text=text,
        score=score,
    )


def test_diversity_and_budget():
    hits = [_hit(1, 0.9), _hit(1, 0.8), _hit(1, 0.7), _hit(1, 0.6), _hit(2, 0.85)]
    packed = pack_context(hits, max_chunks=5, max_tokens=1000, max_per_doc=2, min_score=0.0)
    assert len(packed) == 3  # 2 from doc1 + 1 from doc2
    assert sum(1 for p in packed if p.document_id == 1) == 2
