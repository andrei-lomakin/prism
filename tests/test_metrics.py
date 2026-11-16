from prism.eval.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_precision_mrr():
    relevant = {"a", "b"}
    retrieved = ["x", "a", "b", "y"]
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert mrr_at_k(relevant, retrieved, 10) == 0.5
    assert hit_rate_at_k(relevant, retrieved, 1) == 0.0
    assert hit_rate_at_k(relevant, retrieved, 2) == 1.0


def test_ndcg_perfect():
    relevant = {"a", "b"}
    retrieved = ["a", "b", "c"]
    assert ndcg_at_k(relevant, retrieved, 2) == 1.0
