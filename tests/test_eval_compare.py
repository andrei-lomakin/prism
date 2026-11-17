import json
from pathlib import Path

from prism.eval.compare import (
    collect_runs,
    pair_by_question_id,
    select_baseline,
    to_markdown,
    write_comparison,
)


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
                "n_questions": 10,
                "diagnostic": False,
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
    assert "scientific" in md


def _write_run(
    root: Path,
    *,
    run_id: str,
    eval_config: str,
    retrieval: str,
    recall_by_id: dict[str, float],
    dataset: str = "data/golden/dev.jsonl",
    embedding: str = "emb-a",
    top_k_eval: int = 10,
) -> None:
    d = root / run_id
    d.mkdir(parents=True)
    mean = sum(recall_by_id.values()) / max(1, len(recall_by_id))
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "eval_config": eval_config,
                "dataset": dataset,
                "n_questions": len(recall_by_id),
                "rag": {
                    "chunking": "C2",
                    "embedding": embedding,
                    "retrieval": retrieval,
                    "rewrite": "R0",
                    "rerank": False,
                    "top_k_eval": top_k_eval,
                },
                "retrieval": {"recall@5": {"mean": mean}},
                "ops": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    with (d / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for qid, val in recall_by_id.items():
            f.write(
                json.dumps(
                    {"id": qid, "type": "factoid", "recall@5": val},
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_smoke_not_auto_baseline(tmp_path: Path):
    _write_run(
        tmp_path,
        run_id="2026_smoke",
        eval_config="configs/eval/smoke.yaml",
        retrieval="dense",
        recall_by_id={"q1": 0.0, "q2": 0.0},
    )
    _write_run(
        tmp_path,
        run_id="2026_dense",
        eval_config="configs/eval/ablate_dense.yaml",
        retrieval="dense",
        recall_by_id={"q1": 0.8, "q2": 0.9},
    )
    _write_run(
        tmp_path,
        run_id="2026_hybrid",
        eval_config="configs/eval/ablate_hybrid.yaml",
        retrieval="hybrid",
        recall_by_id={"q1": 1.0, "q2": 1.0},
    )
    rows = collect_runs(tmp_path)
    assert next(r for r in rows if r["run_id"] == "2026_smoke")["diagnostic"] is True
    base = select_baseline(rows)
    assert base is not None
    assert base["run_id"] == "2026_dense"

    out = write_comparison(reports_dir=tmp_path)
    paired = out["paired"]
    assert paired["baseline_run"] == "2026_dense"
    assert paired["best_run"] == "2026_hybrid"
    assert paired["paired_bootstrap"]["n"] == 2
    assert paired["comparability"]["ok"] is True


def test_pair_by_question_id_not_row_index():
    a = {"q_b": 1.0, "q_a": 0.0}
    b = {"q_a": 0.5, "q_b": 0.5, "q_extra": 1.0}
    av, bv, ids = pair_by_question_id(a, b)
    assert ids == ["q_a", "q_b"]
    assert av == [0.0, 1.0]
    assert bv == [0.5, 0.5]


def test_refuse_incomparable_and_mean_only(tmp_path: Path):
    _write_run(
        tmp_path,
        run_id="dense_ok",
        eval_config="configs/eval/ablate_dense.yaml",
        retrieval="dense",
        recall_by_id={"q1": 0.5, "q2": 0.5},
        embedding="emb-a",
    )
    _write_run(
        tmp_path,
        run_id="hybrid_other_emb",
        eval_config="configs/eval/ablate_hybrid.yaml",
        retrieval="hybrid",
        recall_by_id={"q1": 1.0, "q2": 1.0},
        embedding="emb-b",
    )
    out = write_comparison(reports_dir=tmp_path)
    assert out["paired"]["error"] == "incomparable_runs"
    assert out["paired"]["paired_bootstrap"] is None

    # Only means, no overlapping per-question file content mismatch simulated by empty ids
    root = tmp_path / "mean_only"
    root.mkdir()
    _write_run(
        root,
        run_id="d1",
        eval_config="configs/eval/ablate_dense.yaml",
        retrieval="dense",
        recall_by_id={"x1": 0.5, "x2": 0.6},
    )
    _write_run(
        root,
        run_id="h1",
        eval_config="configs/eval/ablate_hybrid.yaml",
        retrieval="hybrid",
        recall_by_id={"y1": 0.9, "y2": 1.0},
    )
    out2 = write_comparison(reports_dir=root)
    assert out2["paired"]["error"] == "insufficient_paired_ids"
    assert out2["paired"]["paired_bootstrap"] is None
