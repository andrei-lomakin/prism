from prism.eval.aggregate import bootstrap_ci, summarize_metric


def test_bootstrap_ci_range():
    vals = [0.0, 0.5, 1.0, 0.5, 0.25]
    m, lo, hi = bootstrap_ci(vals, n_boot=200)
    assert 0.0 <= lo <= m <= hi <= 1.0


def test_summarize_metric():
    s = summarize_metric([1.0, 1.0, 0.0])
    assert s["n"] == 3
    assert 0.0 <= s["mean"] <= 1.0
