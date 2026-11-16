from prism.rag.generate.grounding import check_citations, check_grounding


def test_citations_ok():
    report = check_citations("Ответ [1] и ещё [2].", n_sources=2)
    assert report.ok
    assert report.cited == [1, 2]


def test_citations_missing():
    report = check_citations("Ссылка на [3].", n_sources=2)
    assert not report.ok
    assert report.missing == [3]


def test_grounding_number_in_cited_chunk():
    report = check_grounding(
        "RRF uses k=60 [1].",
        ["Reciprocal Rank Fusion uses constant k = 60."],
    )
    assert report.ok
    assert not report.unsupported_numbers


def test_grounding_flags_missing_number():
    report = check_grounding(
        "The magic number is 99999 [1].",
        ["This chunk has no such figure."],
    )
    assert not report.ok
    assert "99999" in report.unsupported_numbers
