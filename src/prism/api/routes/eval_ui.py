"""Eval dashboard API — list/detail, launch jobs, PDF download."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from prism.api.auth import require_user
from prism.config import get_settings
from prism.eval.compare import collect_runs, is_diagnostic_run, load_run_metrics, write_comparison
from prism.eval.jobs import create_job, list_eval_configs, list_jobs, load_job

router = APIRouter(prefix="/api/eval", tags=["eval"], dependencies=[Depends(require_user)])
ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "reports"

_TS_RE = re.compile(r"^(\d{8}T\d{6}Z)")


class StartEvalRequest(BaseModel):
    eval_config: str = Field(..., min_length=3)
    skip_generation: bool = True


def _fmt_metric(v: Any) -> str | None:
    if v is None:
        return None
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return None


def _run_when(run_id: str) -> str | None:
    m = _TS_RE.match(run_id or "")
    if not m:
        return None
    raw = m.group(1)
    try:
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}:{raw[13:15]}Z"
    except Exception:
        return None


def _short_config(path: Any) -> str:
    s = str(path or "").replace("\\", "/")
    name = s.rsplit("/", 1)[-1]
    return name.replace(".yaml", "").replace(".yml", "") or "—"


def _resolve_run_dir(run_id: str) -> Path:
    direct = REPORTS / run_id
    if direct.is_dir() and (direct / "metrics.json").is_file():
        return direct
    matches = sorted(
        p for p in REPORTS.iterdir() if p.is_dir() and run_id in p.name and (p / "metrics.json").is_file()
    )
    if not matches:
        raise HTTPException(404, "report not found")
    return matches[0]


def _metric_mean(block: dict[str, Any] | None, key: str) -> float | None:
    if not block:
        return None
    m = block.get(key)
    if isinstance(m, dict) and "mean" in m:
        return float(m["mean"])
    if isinstance(m, (int, float)):
        return float(m)
    return None


def _list_items() -> list[dict[str, Any]]:
    rows = collect_runs(REPORTS)
    items: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row.get("run_id") or "")
        role = "diagnostic" if row.get("diagnostic") else "scientific"
        recall = row.get("recall@5")
        retrieval = row.get("retrieval") or "—"
        n_q = row.get("n_questions") or "—"
        folder = REPORTS / rid
        has_pdf = (folder / "report.pdf").is_file()
        items.append(
            {
                "id": rid,
                "title": rid,
                "subtitle": (
                    f"{retrieval} · {_short_config(row.get('eval_config'))} · "
                    f"{n_q} вопр. · Recall@5 {_fmt_metric(recall) or '—'}"
                ),
                "role": role,
                "retrieval": retrieval,
                "n_questions": row.get("n_questions"),
                "recall@5": recall,
                "mrr@10": row.get("mrr@10"),
                "ndcg@10": row.get("ndcg@10"),
                "hit@5": row.get("hit@5"),
                "has_pdf": has_pdf,
                "created_at": _run_when(rid),
            }
        )
    items.sort(key=lambda x: x.get("id") or "", reverse=True)
    return items


def _worst_from_path(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        ret = r.get("retrieval") or {}
        val = ret.get("recall@5")
        if val is None:
            val = 1.0 if ret.get("hit@5") else 0.0
        scored.append((float(val), r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[: max(1, n)]]


def _ensure_pdf(folder: Path, data: dict[str, Any]) -> Path | None:
    pdf = folder / "report.pdf"
    if pdf.is_file():
        return pdf
    per_q: list[dict[str, Any]] = []
    pq = folder / "per_question.jsonl"
    if pq.is_file():
        with pq.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    per_q.append(json.loads(line))
    try:
        from prism.eval.pdf_report import write_pdf_report

        return write_pdf_report(folder, data, per_q)
    except Exception:
        return None


@router.get("/configs")
def get_eval_configs() -> dict[str, Any]:
    return {"items": list_eval_configs()}


@router.get("/jobs")
def get_eval_jobs(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return {"items": list_jobs(limit=limit)}


@router.get("/jobs/{job_id}")
def get_eval_job(job_id: str) -> dict[str, Any]:
    state = load_job(job_id)
    if state is None:
        raise HTTPException(404, "job not found")
    return state


@router.post("/jobs")
async def start_eval_job(body: StartEvalRequest) -> dict[str, Any]:
    """Enqueue eval in arq worker; progress → reports/.jobs/{id}.json."""
    cfg_path = body.eval_config.replace("\\", "/")
    if ".." in cfg_path or not cfg_path.startswith("configs/eval/"):
        raise HTTPException(400, "eval_config must be under configs/eval/")
    full = ROOT / cfg_path
    if not full.is_file():
        raise HTTPException(404, f"config not found: {cfg_path}")

    from prism.eval.jobs import reap_stale_jobs, update_job

    reap_stale_jobs(queued_max_sec=90)
    active = [
        j
        for j in list_jobs(limit=50)
        if j.get("status") in ("queued", "running")
    ]
    if active:
        raise HTTPException(
            409,
            f"Уже идёт прогон {active[0].get('id')} — дождитесь завершения",
        )

    state = create_job(eval_config=cfg_path, skip_generation=body.skip_generation)
    settings = get_settings()
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        await redis.enqueue_job(
            "run_eval_job",
            state["id"],
            cfg_path,
            bool(body.skip_generation),
        )
    except Exception as exc:
        update_job(state["id"], status="failed", error=str(exc), step="enqueue_failed")
        raise HTTPException(503, f"Не удалось поставить в очередь (worker/redis?): {exc}") from exc
    finally:
        await redis.close()
    return state


@router.get("/runs")
def list_eval_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=100),
    refresh: bool = False,
) -> dict[str, Any]:
    if refresh:
        write_comparison(reports_dir=REPORTS)
    items = _list_items()
    total = len(items)
    pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = min(page, pages)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]
    scientific = [it for it in items if it["role"] == "scientific" and it.get("recall@5") is not None]
    pool = scientific or [it for it in items if it.get("recall@5") is not None]
    best = max(pool, key=lambda x: float(x["recall@5"]), default=None) if pool else None
    return {
        "items": page_items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
        "best_run_id": best["id"] if best else None,
        "best_recall@5": best.get("recall@5") if best else None,
    }


@router.get("/runs/{run_id}/pdf")
def download_eval_pdf(run_id: str) -> FileResponse:
    folder = _resolve_run_dir(run_id)
    data = load_run_metrics(folder)
    if not data:
        raise HTTPException(404, "metrics.json missing")
    pdf = _ensure_pdf(folder, data)
    if pdf is None or not pdf.is_file():
        raise HTTPException(
            503,
            "PDF недоступен (нужен fpdf2 и TTF с кириллицей, напр. Segoe UI / DejaVu)",
        )
    return FileResponse(
        path=pdf,
        media_type="application/pdf",
        filename=f"{folder.name}.pdf",
    )


@router.get("/runs/{run_id}")
def get_eval_run(run_id: str, worst_n: int = Query(5, ge=1, le=50)) -> dict[str, Any]:
    folder = _resolve_run_dir(run_id)
    data = load_run_metrics(folder)
    if not data:
        raise HTTPException(404, "metrics.json missing")
    diagnostic = is_diagnostic_run(data, dir_name=folder.name)
    ret = data.get("retrieval") or {}
    gen = data.get("generation") or {}
    ops = data.get("ops") or {}
    rag = data.get("rag") or {}
    summary_path = folder / "summary.md"
    worst = _worst_from_path(folder / "per_question.jsonl", worst_n)
    has_pdf = (folder / "report.pdf").is_file()
    return {
        "id": data.get("run_id") or folder.name,
        "dir": folder.name,
        "role": "diagnostic" if diagnostic else "scientific",
        "eval_config": data.get("eval_config"),
        "rag_config": data.get("rag_config"),
        "dataset": data.get("dataset"),
        "n_questions": data.get("n_questions"),
        "created_at": _run_when(str(data.get("run_id") or folder.name)),
        "rag": rag,
        "metrics": {
            "recall@5": _metric_mean(ret, "recall@5"),
            "mrr@10": _metric_mean(ret, "mrr@10"),
            "ndcg@10": _metric_mean(ret, "ndcg@10"),
            "hit@5": _metric_mean(ret, "hit@5"),
            "precision@5": _metric_mean(ret, "precision@5"),
            "context_precision": _metric_mean(ret, "context_precision"),
            "context_recall": _metric_mean(ret, "context_recall"),
            "faithfulness": _metric_mean(gen, "faithfulness"),
            "answer_relevance": _metric_mean(gen, "answer_relevance"),
            "retrieve_p95_ms": ops.get("retrieve_p95_ms"),
            "usd_per_1k": ops.get("usd_per_1k"),
        },
        "worst": worst,
        "summary_md": summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "",
        "has_pdf": has_pdf,
        "raw": data,
    }


@router.get("/ablations")
def list_ablations(refresh: bool = False) -> dict[str, Any]:
    if refresh:
        write_comparison(reports_dir=REPORTS)
    rows = collect_runs(REPORTS)
    md_path = REPORTS / "ABLATION_TABLE.md"
    return {
        "n": len(rows),
        "rows": rows,
        "markdown": md_path.read_text(encoding="utf-8") if md_path.is_file() else "",
    }


@router.get("/runs/{run_id}/worst")
def worst_questions(run_id: str, n: int = 2) -> dict[str, Any]:
    folder = _resolve_run_dir(run_id)
    path = folder / "per_question.jsonl"
    if not path.is_file():
        raise HTTPException(404, "per_question.jsonl missing")
    return {"run_id": folder.name, "worst": _worst_from_path(path, n)}
