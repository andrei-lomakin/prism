from prism.eval.compare import to_markdown


def test_to_markdown_empty():
    md = to_markdown([])
    assert "Recall@5" in md


def test_to_markdown_row():
    md = to_markdown(
        [
            {
                "run_id": "r1",
                "retrieval": "hybrid",
                "chunking": "C2",
                "recall@5": 0.9,
                "mrr@10": 0.7,
                "ndcg@10": 0.8,
                "hit@5": 1.0,
                "faithfulness": None,
                "answer_relevance": None,
                "retrieve_p95_ms": 100.0,
                "usd_per_1k": 0.0,
            }
        ]
    )
    assert "0.900" in md
    assert "hybrid" in md
