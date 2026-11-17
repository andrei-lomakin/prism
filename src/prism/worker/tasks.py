"""arq task functions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prism.db.session import get_session_factory
from prism.eval.jobs import update_job
from prism.ingest.pipeline import run_ingest

logger = logging.getLogger(__name__)


async def ingest_document(ctx: dict, job_id: int, force_reindex: bool = False) -> None:
    """Parse → chunk → embed → save for a queued Job row."""
    db = get_session_factory()()
    try:
        run_ingest(db, job_id, force_reindex=bool(force_reindex))
    finally:
        db.close()


async def run_eval_job(
    ctx: dict,
    job_id: str,
    eval_config: str,
    skip_generation: bool = True,
) -> dict[str, Any]:
    """Run offline eval with live progress written to reports/.jobs/{id}.json."""
    from prism.eval.compare import write_comparison
    from prism.eval.runner import run_eval

    update_job(job_id, status="running", step="starting", progress=1, detail=eval_config)

    def on_progress(payload: dict[str, Any]) -> None:
        fields = {
            "status": "running",
            "step": payload.get("step"),
            "detail": payload.get("detail") or payload.get("question") or "",
            "progress": int(payload.get("progress") or 0),
            "i": payload.get("i") or 0,
            "n": payload.get("n") or 0,
            "question_id": payload.get("question_id"),
            "question": payload.get("question"),
            "run_id": payload.get("run_id"),
        }
        update_job(job_id, **{k: v for k, v in fields.items() if v is not None})

    try:
        summary = await asyncio.to_thread(
            run_eval,
            eval_config,
            skip_generation=bool(skip_generation),
            on_progress=on_progress,
        )
        try:
            write_comparison()
        except Exception:
            logger.exception("write_comparison after eval failed")
        update_job(
            job_id,
            status="completed",
            step="done",
            progress=100,
            run_id=summary.get("run_id"),
            detail=summary.get("report_dir") or "",
            error=None,
        )
        return {"run_id": summary.get("run_id"), "job_id": job_id}
    except Exception as exc:
        logger.exception("eval job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            step="error",
            error=str(exc),
            detail=str(exc),
        )
        raise
