from prism.eval.compare import recall_bar_svg


def test_recall_bar_svg_empty():
    svg = recall_bar_svg([])
    assert "svg" in svg


def test_recall_bar_svg_bars():
    svg = recall_bar_svg(
        [
            {"run_id": "a", "recall@5": 0.9},
            {"run_id": "b", "recall@5": 0.5},
        ]
    )
    assert "0.900" in svg or "0.9" in svg
    assert "rect" in svg
