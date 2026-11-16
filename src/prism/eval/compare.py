"""Build comparative ablation table from reports/*/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism.eval.stats import paired_bootstrap

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
        prov = data.get("provenance") or {}
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
                "commit": prov.get("commit_short"),
                "_dir": str(child),
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
        "commit",
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
                    str(r.get("commit") or "—"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _best_vs_baseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [r for r in rows if r.get("recall@5") is not None]
    if len(scored) < 2:
        return None
    best = max(scored, key=lambda r: float(r["recall@5"]))
    base = next((r for r in scored if r.get("retrieval") == "dense"), None)
    if base is None or base["run_id"] == best["run_id"]:
        # second best as baseline
        ordered = sorted(scored, key=lambda r: float(r["recall@5"]), reverse=True)
        if len(ordered) < 2:
            return None
        best, base = ordered[0], ordered[1]

    # Use means as single-point paired bootstrap of constant vectors (honest: n=1)
    # Prefer per_question if available
    def series(run: dict[str, Any], key: str) -> list[float]:
        d = Path(str(run.get("_dir") or ""))
        pq = d / "per_question.jsonl"
        vals: list[float] = []
        if pq.is_file():
            with pq.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("type") == "unanswerable":
                        continue
                    v = row.get(key)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
        if not vals and run.get(key) is not None:
            vals = [float(run[key])]
        return vals

    a = series(best, "recall@5")
    b = series(base, "recall@5")
    # Align by min length if different question counts
    n = min(len(a), len(b))
    paired = paired_bootstrap(a[:n], b[:n]) if n else None
    return {
        "best_run": best.get("run_id"),
        "baseline_run": base.get("run_id"),
        "metric": "recall@5",
        "paired_bootstrap": paired,
    }


def recall_bar_svg(rows: list[dict[str, Any]], *, width: int = 640, height: int = 280) -> str:
    scored = [r for r in rows if r.get("recall@5") is not None][:12]
    if not scored:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='40'><text x='8' y='24'>no data</text></svg>"
    margin = 80
    bar_h = max(12, (height - 40) // max(1, len(scored)) - 4)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='#fffdf8'/>",
        "<text x='8' y='18' font-size='12' fill='#1c1a17'>Recall@5 by run</text>",
    ]
    max_w = width - margin - 20
    for i, r in enumerate(scored):
        y = 30 + i * (bar_h + 4)
        val = float(r["recall@5"])
        w = max(1, int(max_w * val))
        label = str(r.get("run_id") or "")[:22]
        parts.append(
            f"<text x='8' y='{y + bar_h - 2}' font-size='10' fill='#6b6560'>{label}</text>"
        )
        parts.append(
            f"<rect x='{margin}' y='{y}' width='{w}' height='{bar_h}' fill='#0f6b5c' rx='2'/>"
        )
        parts.append(
            f"<text x='{margin + w + 4}' y='{y + bar_h - 2}' font-size='10' fill='#1c1a17'>{val:.3f}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_comparison(
    *,
    reports_dir: Path | None = None,
    out_md: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    rows = collect_runs(reports_dir)
    md = to_markdown(rows)
    paired = _best_vs_baseline(rows)
    if paired:
        md += "\n## Best vs baseline (paired bootstrap on recall@5)\n\n"
        md += "```json\n" + json.dumps(paired, ensure_ascii=False, indent=2) + "\n```\n"
    out_dir = reports_dir or REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_md or (out_dir / "ABLATION_TABLE.md")
    json_path = out_json or (out_dir / "ABLATION_TABLE.json")
    svg_path = out_dir / "ABLATION_RECALL.svg"
    md_path.write_text(md, encoding="utf-8")
    payload = {"rows": rows, "paired_best_vs_baseline": paired}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    svg_path.write_text(recall_bar_svg(rows), encoding="utf-8")
    return {
        "n_runs": len(rows),
        "markdown": str(md_path),
        "json": str(json_path),
        "svg": str(svg_path),
        "paired": paired,
        "rows": rows,
    }
