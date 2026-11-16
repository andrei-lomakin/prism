from prism.eval.kappa_tools import kappa_from_files
from prism.eval.metrics import answer_correctness_lexical
from prism.eval.stats import cohens_kappa, paired_bootstrap


def test_correctness_f1():
    assert answer_correctness_lexical("hello world test", "hello world") > 0.5


def test_paired_bootstrap_identical():
    xs = [0.9, 0.8, 0.7]
    out = paired_bootstrap(xs, xs)
    assert out["delta"] == 0.0
    assert out["n"] == 3


def test_cohens_kappa_perfect():
    labs = ["a", "b", "a", "b"]
    assert cohens_kappa(labs, labs) == 1.0


def test_kappa_from_example_files():
    out = kappa_from_files(
        "data/golden/annotator_a.example.jsonl",
        "data/golden/annotator_b.example.jsonl",
    )
    assert out["n_overlap"] == 5
    assert out["kappa"] is not None
    assert 0.0 <= out["kappa"] <= 1.0
