"""Eval dashboard API — serves aggregated ablation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from prism.api.auth import require_user
from prism.eval.compare import collect_runs, recall_bar_svg, write_comparison

router = APIRouter(prefix="/api/eval", tags=["eval"], dependencies=[Depends(require_user)])
ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "reports"


@router.get("/ablations")
def list_ablations(refresh: bool = False) -> dict[str, Any]:
    if refresh:
        write_comparison(reports_dir=REPORTS)
    rows = collect_runs(REPORTS)
    md_path = REPORTS / "ABLATION_TABLE.md"
    svg_path = REPORTS / "ABLATION_RECALL.svg"
    return {
        "n": len(rows),
        "rows": rows,
        "markdown": md_path.read_text(encoding="utf-8") if md_path.is_file() else "",
        "svg": svg_path.read_text(encoding="utf-8") if svg_path.is_file() else recall_bar_svg(rows),
    }


@router.get("/runs/{run_id}/worst")
def worst_questions(run_id: str, n: int = 2) -> dict[str, Any]:
    """Pick lowest recall@5 questions from a report's per_question.jsonl."""
    folder = REPORTS / run_id
    path = folder / "per_question.jsonl"
    if not path.is_file():
        # fuzzy prefix match
        matches = [p for p in REPORTS.iterdir() if p.is_dir() and run_id in p.name]
        if not matches:
            raise HTTPException(404, "report not found")
        path = matches[0] / "per_question.jsonl"
    if not path.is_file():
        raise HTTPException(404, "per_question.jsonl missing")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    scored = []
    for r in rows:
        ret = r.get("retrieval") or {}
        # prefer recall@5 if present as float, else hit
        val = ret.get("recall@5")
        if val is None:
            val = 1.0 if ret.get("hit@5") else 0.0
        scored.append((float(val), r))
    scored.sort(key=lambda x: x[0])
    worst = [r for _, r in scored[: max(1, n)]]
    return {"run_id": run_id, "worst": worst}
