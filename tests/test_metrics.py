from prism.eval.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_precision_mrr():
    relevant = {"a", "b"}
    retrieved = ["x", "a", "b", "y"]
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert mrr_at_k(relevant, retrieved, 10) == 0.5
    assert hit_rate_at_k(relevant, retrieved, 1) == 0.0
    assert hit_rate_at_k(relevant, retrieved, 2) == 1.0


def test_precision_at_k_uses_k_not_list_len():
    """Short retrieval must not inflate P@k (1 relevant of requested 5 → 0.2)."""
    relevant = {"a"}
    retrieved = ["a"]
    assert precision_at_k(relevant, retrieved, 5) == 0.2
    assert precision_at_k(relevant, [], 5) == 0.0


def test_ndcg_perfect():
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert ndcg_at_k(relevant, retrieved, 2) == 1.0


def test_ndcg_idcg_uses_missing_gold():
    """Partial hit must not score 1.0 when other gold docs exist."""
    import math

    relevant = {"a", "b", "c"}
    retrieved = ["a", "x", "y"]
    # Old bug: ideal = sort([1,0,0]) → nDCG=1.0. Correct ideal = [1,1,1].
    got = ndcg_at_k(relevant, retrieved, 3)
    dcg_fact = 1.0  # only pos1 relevant: (2^1-1)/log2(2)
    dcg_ideal = (
        1.0 / math.log2(2)
        + 1.0 / math.log2(3)
        + 1.0 / math.log2(4)
    )
    assert abs(got - dcg_fact / dcg_ideal) < 1e-9
    assert got < 1.0


def test_context_precision_not_precision_at_k():
    from prism.eval.metrics import context_precision_at_k

    # Early relevant → CP high; P@k still /k
    labels = [True, False, False, False, False]
    assert context_precision_at_k(labels, 5) == 1.0
    assert precision_at_k({"a"}, ["a", "x", "y", "z", "w"], 5) == 0.2

    # Relevant only at rank 2 → CP = precision@2 = 0.5
    assert context_precision_at_k([False, True], 2) == 0.5


def test_context_recall_section_aware():
    from prism.eval.metrics import context_recall_at_k

    gold = [("doc1", "Intro"), ("doc2", None)]
    # Wrong section on doc1 — unit not covered; doc2 covered by any section
    hits = [("doc1", "Appendix"), ("doc2", "Anywhere")]
    assert context_recall_at_k(gold, hits, 2) == 0.5
    # Soft substring match
    hits2 = [("doc1", "Chapter / Intro / more"), ("doc2", "x")]
    assert context_recall_at_k(gold, hits2, 2) == 1.0
