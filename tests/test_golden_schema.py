from collections import defaultdict

from prism.eval.golden_build import (
    _controls,
    _seeds,
    assert_no_family_leak,
    assign_splits,
    build,
    expand_seeds,
    family_of,
)
from prism.eval.schema import GoldenQuestion, RelevantDoc


def test_seeds_consistency():
    for seed in _seeds() + _controls():
        q = GoldenQuestion(
            id=seed["key"],
            question=seed["question"],
            language=seed["language"],
            type=seed["type"],
            relevant_docs=seed["docs"],
            gold_answer=seed["gold_answer"],
            split="dev",
            reviewed=seed.get("reviewed", False),
            family=family_of(seed["key"]),
        )
        q.validate_consistency()


def test_family_of_cognates():
    assert family_of("f_rag_def_en") == family_of("f_rag_def_ru") == "f_rag_def"
    assert family_of("ctrl_doc_key_en") == "ctrl_doc_key"


def test_build_distribution():
    summary = build(target=240)
    assert summary["total"] == 240
    assert summary["all"]["unanswerable_pct"] >= 10
    assert summary["all"]["multihop_pct"] >= 15
    assert summary["all"]["reviewed"] >= 30
    assert summary["dev"]["n"] + summary["test"]["n"] == 240
    assert summary["test"]["n"] >= 40
    assert summary["split_unit"] == "family"
    assert summary["controls"] >= 4


def test_no_family_leak_across_splits():
    items = expand_seeds(_seeds(), target=120, controls=_controls())
    assign_splits(items)
    assert_no_family_leak(items)
    by_fam: dict[str, set[str]] = defaultdict(set)
    for it in items:
        by_fam[it.family].add(it.split)
    assert all(len(sp) == 1 for sp in by_fam.values())
    assert {i.split for i in items} == {"dev", "test"}


def test_paraphrase_dedupes_identical_text():
    items = expand_seeds(_seeds(), target=240, controls=_controls())
    norms = [re_norm(i.question) for i in items]
    assert len(norms) == len(set(norms))


def re_norm(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text.strip().lower())


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
    items = expand_seeds(_seeds(), target=80, controls=_controls())
    assign_splits(items)
    assert {i.split for i in items} == {"dev", "test"}
