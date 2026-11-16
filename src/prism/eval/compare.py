"""Build comparative ablation table from reports/*/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"


def _metric(block: dict[str, Any] | None, key: str) -> float | None:
    if not block:
        return None
    m = block.get(key) or {}
    if isinstance(m, dict) and "mean" in m:
        return float(m["mean"])
    if isinstance(m, (int, float)):
        return float(m)
    return None


def load_run_metrics(path: Path) -> dict[str, Any] | None:
    metrics = path / "metrics.json"
    if not metrics.is_file():
        return None
    return json.loads(metrics.read_text(encoding="utf-8"))


def collect_runs(reports_dir: Path | None = None) -> list[dict[str, Any]]:
    root = reports_dir or REPORTS
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        data = load_run_metrics(child)
        if not data:
            continue
        ret = data.get("retrieval") or {}
        ops = data.get("ops") or {}
        gen = data.get("generation") or {}
        rag = data.get("rag") or {}
        rows.append(
            {
                "run_id": data.get("run_id") or child.name,
                "eval_config": data.get("eval_config"),
                "rag_config": data.get("rag_config"),
                "dataset": data.get("dataset"),
                "n_questions": data.get("n_questions"),
                "chunking": rag.get("chunking"),
                "embedding": rag.get("embedding"),
                "retrieval": rag.get("retrieval"),
                "recall@5": _metric(ret, "recall@5"),
                "mrr@10": _metric(ret, "mrr@10"),
                "ndcg@10": _metric(ret, "ndcg@10"),
                "hit@5": _metric(ret, "hit@5"),
                "faithfulness": _metric(gen, "faithfulness"),
                "answer_relevance": _metric(gen, "answer_relevance"),
                "retrieve_p95_ms": ops.get("retrieve_p95_ms"),
                "usd_per_1k": ops.get("usd_per_1k"),
            }
        )
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run_id",
        "retrieval",
        "chunking",
        "Recall@5",
        "MRR@10",
        "nDCG@10",
        "Hit@5",
        "Faith.",
        "AnsRel",
        "p95_ms",
        "$/1k",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    def fmt(v: Any, nd: int = 3) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("run_id") or "")[:28],
                    str(r.get("retrieval") or "—"),
                    str(r.get("chunking") or "—"),
                    fmt(r.get("recall@5")),
                    fmt(r.get("mrr@10")),
                    fmt(r.get("ndcg@10")),
                    fmt(r.get("hit@5")),
                    fmt(r.get("faithfulness")),
                    fmt(r.get("answer_relevance")),
                    fmt(r.get("retrieve_p95_ms"), 1),
                    fmt(r.get("usd_per_1k"), 4),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_comparison(
    *,
    reports_dir: Path | None = None,
    out_md: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    rows = collect_runs(reports_dir)
    md = to_markdown(rows)
    out_dir = reports_dir or REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_md or (out_dir / "ABLATION_TABLE.md")
    json_path = out_json or (out_dir / "ABLATION_TABLE.json")
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n_runs": len(rows), "markdown": str(md_path), "json": str(json_path), "rows": rows}
