"""Build comparative ablation table from reports/*/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism.eval.stats import paired_bootstrap

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"

# Factors that must match for a scientific paired compare (ablate only the study factor).
COMPARABILITY_FIELDS = (
    "dataset",
    "chunking",
    "embedding",
    "rewrite",
    "rerank",
    "top_k_eval",
)

# Preferred scientific baselines (first match wins among non-diagnostic runs).
DEFAULT_BASELINE_CONFIGS = (
    "configs/eval/ablate_dense.yaml",
    "configs/eval/baseline.yaml",
    "configs/eval/baseline_test.yaml",
)


def _metric(block: dict[str, Any] | None, key: str) -> float | None:
    if not block:
        return None
    m = block.get(key) or {}
    if isinstance(m, dict) and "mean" in m:
        return float(m["mean"])
    if isinstance(m, (int, float)):
        return float(m)
    return None


def _norm_path(p: Any) -> str:
    return str(p or "").replace("\\", "/")


def is_diagnostic_run(data: dict[str, Any], *, dir_name: str = "") -> bool:
    """Smoke / broken-label / explicitly flagged runs — keep in table, never as baseline."""
    if data.get("diagnostic") is True:
        return True
    cfg = _norm_path(data.get("eval_config")).lower()
    rid = str(data.get("run_id") or dir_name).lower()
    if "smoke" in cfg or rid.endswith("_smoke") or "/smoke.yaml" in cfg:
        return True
    notes = str(data.get("diagnostic_note") or "").strip()
    return bool(notes)


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
        diagnostic = is_diagnostic_run(data, dir_name=child.name)
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
                "rewrite": rag.get("rewrite"),
                "rerank": rag.get("rerank"),
                "top_k_eval": rag.get("top_k_eval"),
                "diagnostic": diagnostic,
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
        "role",
        "retrieval",
        "chunking",
        "n_q",
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
        role = "diagnostic" if r.get("diagnostic") else "scientific"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("run_id") or "")[:28],
                    role,
                    str(r.get("retrieval") or "—"),
                    str(r.get("chunking") or "—"),
                    str(r.get("n_questions") or "—"),
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


def select_baseline(
    rows: list[dict[str, Any]],
    *,
    baseline_run: str | None = None,
    baseline_config: str | None = None,
) -> dict[str, Any] | None:
    """Pick scientific baseline. Never falls back to smoke / diagnostic."""
    by_id = {str(r.get("run_id")): r for r in rows}
    if baseline_run:
        hit = by_id.get(baseline_run)
        if hit is None:
            return None
        return hit

    scientific = [r for r in rows if not r.get("diagnostic") and r.get("recall@5") is not None]
    if not scientific:
        return None

    needles: list[str] = []
    if baseline_config:
        needles.append(_norm_path(baseline_config))
    needles.extend(DEFAULT_BASELINE_CONFIGS)

    for needle in needles:
        matches = [
            r for r in scientific if needle.lower() in _norm_path(r.get("eval_config")).lower()
        ]
        if matches:
            matches.sort(
                key=lambda r: (int(r.get("n_questions") or 0), str(r.get("run_id") or "")),
                reverse=True,
            )
            return matches[0]

    # Last resort: non-diagnostic dense only — never promote hybrid/other as "baseline".
    dense = [r for r in scientific if r.get("retrieval") == "dense"]
    if not dense:
        return None
    dense.sort(
        key=lambda r: (int(r.get("n_questions") or 0), str(r.get("run_id") or "")),
        reverse=True,
    )
    return dense[0]


def comparability(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for field in COMPARABILITY_FIELDS:
        va, vb = a.get(field), b.get(field)
        if va is None or vb is None:
            continue
        if _norm_path(va) != _norm_path(vb):
            mismatches.append({"field": field, "a": va, "b": vb})
    return {"ok": not mismatches, "mismatches": mismatches}


def load_per_question_series(run: dict[str, Any], key: str) -> dict[str, float]:
    """Map question_id → metric. Empty if per_question.jsonl missing."""
    d = Path(str(run.get("_dir") or ""))
    pq = d / "per_question.jsonl"
    out: dict[str, float] = {}
    if not pq.is_file():
        return out
    with pq.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "unanswerable":
                continue
            qid = row.get("id")
            v = row.get(key)
            if qid is None or not isinstance(v, (int, float)):
                continue
            out[str(qid)] = float(v)
    return out


def pair_by_question_id(
    a: dict[str, float],
    b: dict[str, float],
) -> tuple[list[float], list[float], list[str]]:
    ids = sorted(set(a) & set(b))
    return [a[i] for i in ids], [b[i] for i in ids], ids


def _best_vs_baseline(
    rows: list[dict[str, Any]],
    *,
    baseline_run: str | None = None,
    baseline_config: str | None = None,
    metric: str = "recall@5",
) -> dict[str, Any] | None:
    base = select_baseline(rows, baseline_run=baseline_run, baseline_config=baseline_config)
    if base is None:
        return {
            "error": "no_scientific_baseline",
            "hint": "Pass --baseline RUN_ID or run configs/eval/ablate_dense.yaml / baseline.yaml",
        }

    scientific = [
        r
        for r in rows
        if not r.get("diagnostic")
        and r.get(metric) is not None
        and r.get("run_id") != base.get("run_id")
    ]
    if not scientific:
        return {
            "baseline_run": base.get("run_id"),
            "error": "no_candidate_vs_baseline",
        }

    # Prefer highest metric among runs comparable to baseline; else best overall + warn.
    comparable = [r for r in scientific if comparability(base, r)["ok"]]
    pool = comparable or scientific
    best = max(pool, key=lambda r: float(r[metric]))
    compat = comparability(base, best)

    a_map = load_per_question_series(best, metric)
    b_map = load_per_question_series(base, metric)
    a_vals, b_vals, qids = pair_by_question_id(a_map, b_map)

    result: dict[str, Any] = {
        "best_run": best.get("run_id"),
        "baseline_run": base.get("run_id"),
        "baseline_eval_config": base.get("eval_config"),
        "metric": metric,
        "comparability": compat,
        "n_paired_questions": len(qids),
        "n_only_best": len(set(a_map) - set(b_map)),
        "n_only_baseline": len(set(b_map) - set(a_map)),
    }

    if not compat["ok"]:
        result["paired_bootstrap"] = None
        result["error"] = "incomparable_runs"
        result["note"] = (
            "Paired CI refused: dataset/chunking/embedding/rewrite/rerank/top_k_eval differ. "
            "Ablate one factor at a time."
        )
        return result

    if len(qids) < 2:
        result["paired_bootstrap"] = None
        result["error"] = "insufficient_paired_ids"
        result["note"] = (
            "Need ≥2 overlapping question_ids in per_question.jsonl. "
            "Refusing mean-only / index-aligned bootstrap."
        )
        return result

    result["paired_bootstrap"] = paired_bootstrap(a_vals, b_vals)
    result["sample_question_ids"] = qids[:5]
    return result


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
        fill = "#a67c52" if r.get("diagnostic") else "#0f6b5c"
        parts.append(
            f"<text x='8' y='{y + bar_h - 2}' font-size='10' fill='#6b6560'>{label}</text>"
        )
        parts.append(
            f"<rect x='{margin}' y='{y}' width='{w}' height='{bar_h}' fill='{fill}' rx='2'/>"
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
    baseline_run: str | None = None,
    baseline_config: str | None = None,
) -> dict[str, Any]:
    rows = collect_runs(reports_dir)
    md = to_markdown(rows)
    paired = _best_vs_baseline(
        rows,
        baseline_run=baseline_run,
        baseline_config=baseline_config,
    )
    if paired:
        md += "\n## Best vs baseline (paired bootstrap on recall@5)\n\n"
        md += (
            "Baseline: explicit `--baseline` / `--baseline-config`, else "
            "`ablate_dense` -> `baseline.yaml` (smoke/diagnostic excluded).\n"
            "Pairs join on `question_id`; CI requires comparable controls + >=2 shared ids.\n\n"
        )
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
