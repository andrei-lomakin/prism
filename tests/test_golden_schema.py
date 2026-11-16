from prism.eval.golden_build import assign_splits, build, expand_seeds, _seeds
from prism.eval.schema import GoldenQuestion, RelevantDoc


def test_seeds_consistency():
    for seed in _seeds():
        q = GoldenQuestion(
            id=seed["key"],
            question=seed["question"],
            language=seed["language"],
            type=seed["type"],
            relevant_docs=seed["docs"],
            gold_answer=seed["gold_answer"],
            split="dev",
            reviewed=seed.get("reviewed", False),
        )
        q.validate_consistency()


def test_build_distribution():
    summary = build(target=240)
    assert summary["total"] == 240
    assert summary["all"]["unanswerable_pct"] >= 10
    assert summary["all"]["multihop_pct"] >= 15
    assert summary["all"]["reviewed"] >= 30
    assert summary["dev"]["n"] + summary["test"]["n"] == 240
    assert summary["test"]["n"] >= 50


def test_unanswerable_rejects_docs():
    q = GoldenQuestion(
        id="bad",
        question="What is the weather tomorrow?",
        language="en",
        type="unanswerable",
        relevant_docs=[RelevantDoc(doc_key="01_rag_overview_en")],
        gold_answer="none",
        split="dev",
    )
    try:
        q.validate_consistency()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_split_assigns_both():
    items = expand_seeds(_seeds(), target=80)
    assign_splits(items)
    assert {i.split for i in items} == {"dev", "test"}
