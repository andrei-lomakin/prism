from prism.rag.generate.grounding import check_citations


def test_citations_ok():
    report = check_citations("Ответ [1] и ещё [2].", n_sources=2)
    assert report.ok
    assert report.cited == [1, 2]


def test_citations_missing():
    report = check_citations("Ссылка на [3].", n_sources=2)
    assert not report.ok
    assert report.missing == [3]
