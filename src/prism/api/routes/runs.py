"""Agent runs, traces, and approval queue."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from prism.agents.policy import get_approval_store
from prism.agents.policy.approvals import ApprovalRecord
from prism.api.auth import require_user
from prism.api.settings_store import settings_as_config
from prism.db.models import AgentRun, Step
from prism.db.session import get_db
from prism.observability import trace_buffer

router = APIRouter(prefix="/api", tags=["runs"], dependencies=[Depends(require_user)])
logger = logging.getLogger(__name__)


def _coerce_run_id(run_id: str | int | None) -> int | None:
    if isinstance(run_id, int):
        return run_id
    if isinstance(run_id, str) and run_id.isdigit():
        return int(run_id)
    return None


def _resume_approved(db: Session, rec: ApprovalRecord) -> dict[str, Any] | None:
    """Continue the run that asked for this approval. Same run id, original args."""
    run_id = _coerce_run_id(rec.run_id)
    if run_id is None:
        return None
    row = db.get(AgentRun, run_id)
    if row is None or row.state != "awaiting_approval":
        return None
    from prism.agents import AgentRuntime

    runtime = AgentRuntime(db, settings=settings_as_config(db), persist=True)
    answer = runtime.resume(run_id)
    return {"run_id": answer.run_id, "state": answer.state, "answer": answer.answer}


class ApprovalDecision(BaseModel):
    reason: str | None = None


@router.get("/runs")
def list_runs(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Paginated agent runs (newest first). Top-level + children included."""
    total = db.query(func.count(AgentRun.id)).scalar() or 0
    rows = (
        db.query(AgentRun)
        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    step_counts = dict(db.query(Step.run_id, func.count(Step.id)).group_by(Step.run_id).all())
    items = []
    for row in rows:
        n_steps = int(step_counts.get(row.id) or 0)
        objective = (row.objective or "").strip().replace("\n", " ")
        items.append(
            {
                "kind": "run",
                "id": str(row.id),
                "title": objective[:120] + ("…" if len(objective) > 120 else ""),
                "subtitle": f"{row.role} · {n_steps} шаг(ов)",
                "state": row.state,
                "role": row.role,
                "parent_id": row.parent_id,
                "n_steps": n_steps,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": int(total),
        "pages": max(1, (int(total) + page_size - 1) // page_size),
    }


@router.get("/pipeline-logs")
def list_pipeline_logs(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """Unified journal: persisted agent runs + recent in-memory RAG traces."""
    run_total = int(db.query(func.count(AgentRun.id)).scalar() or 0)
    run_rows = (
        db.query(AgentRun).order_by(AgentRun.created_at.desc(), AgentRun.id.desc()).limit(500).all()
    )
    step_counts = dict(db.query(Step.run_id, func.count(Step.id)).group_by(Step.run_id).all())
    items: list[dict[str, Any]] = []
    run_ids = set()
    for row in run_rows:
        run_ids.add(str(row.id))
        n_steps = int(step_counts.get(row.id) or 0)
        objective = (row.objective or "").strip().replace("\n", " ")
        items.append(
            {
                "kind": "run",
                "id": str(row.id),
                "title": objective[:120] + ("…" if len(objective) > 120 else ""),
                "subtitle": f"агент · {row.role} · {n_steps} шаг(ов)",
                "state": row.state,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "sort_ms": row.created_at.timestamp() * 1000 if row.created_at else 0.0,
            }
        )
    for tr in trace_buffer.list_summaries():
        if tr["id"] in run_ids:
            continue
        items.append(
            {
                **tr,
                "subtitle": f"поиск/RAG · {tr.get('n_spans', 0)} span(ов)",
                "created_at": None,
                "sort_ms": float(tr.get("started_ms") or 0.0),
            }
        )
    items.sort(key=lambda x: float(x.get("sort_ms") or 0.0), reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]
    for it in page_items:
        it.pop("sort_ms", None)
        it.pop("started_ms", None)
        it.pop("n_spans", None)
    return {
        "items": page_items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": max(1, (total + page_size - 1) // page_size),
        "runs_in_db": run_total,
    }


@router.delete("/pipeline-logs")
def clear_pipeline_logs(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Wipe journal: persisted agent runs/steps + in-memory RAG traces."""
    n_steps = db.query(Step).delete()
    # Break self-FK before bulk delete (parent_id → agent_runs.id).
    db.query(AgentRun).update({AgentRun.parent_id: None}, synchronize_session=False)
    n_runs = db.query(AgentRun).delete()
    db.commit()
    n_traces = len(trace_buffer.list_summaries())
    trace_buffer.clear()
    return {
        "ok": True,
        "deleted_runs": int(n_runs or 0),
        "deleted_steps": int(n_steps or 0),
        "cleared_traces": int(n_traces),
    }


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    row = db.get(AgentRun, run_id)
    if row is None:
        raise HTTPException(404, "run not found")
    steps = db.query(Step).filter(Step.run_id == run_id).order_by(Step.n.asc()).all()
    return {
        "id": row.id,
        "parent_id": row.parent_id,
        "role": row.role,
        "objective": row.objective,
        "state": row.state,
        "plan": row.plan,
        "budget": row.budget,
        "usage": row.usage,
        "result": row.result,
        "steps": [
            {
                "n": s.n,
                "kind": s.kind,
                "input": s.input,
                "output": s.output,
                "tokens": s.tokens,
                "cost": s.cost,
                "latency_ms": s.latency_ms,
                "span_id": s.span_id,
            }
            for s in steps
        ],
    }


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    spans = trace_buffer.get(trace_id)
    # Also accept run_id as int string → try buffer + empty
    return {"trace_id": trace_id, "spans": spans, "n": len(spans)}


@router.get("/approvals")
def list_approvals() -> dict[str, Any]:
    store = get_approval_store()
    items = store.pending()
    return {
        "items": [
            {
                "id": a.id,
                "run_id": a.run_id,
                "tool": a.tool,
                "args": a.args,
                "rule_id": a.rule_id,
                "created_at": a.created_at,
                "expires_at": a.expires_at,
                "status": a.status,
            }
            for a in items
        ]
    }


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Continue a run that is already awaiting_approval (action must be approved)."""
    row = db.get(AgentRun, run_id)
    if row is None:
        raise HTTPException(404, "run not found")
    if row.state != "awaiting_approval":
        raise HTTPException(409, f"run state is {row.state}")
    from prism.agents import AgentRuntime

    runtime = AgentRuntime(db, settings=settings_as_config(db), persist=True)
    try:
        answer = runtime.resume(run_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"run_id": answer.run_id, "state": answer.state, "answer": answer.answer}


@router.post("/approvals/{approval_id}/approve")
def approve(
    approval_id: str,
    body: ApprovalDecision | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    store = get_approval_store()
    rec = store.approve(approval_id, reason=(body.reason if body else None))
    if rec is None:
        raise HTTPException(404, "approval not found")
    resumed: dict[str, Any] | None = None
    resume_error: str | None = None
    try:
        resumed = _resume_approved(db, rec)
    except Exception as exc:  # noqa: BLE001 — approval itself already stuck
        logger.exception("resume after approve failed id=%s", approval_id)
        resume_error = str(exc)
    out: dict[str, Any] = {"id": rec.id, "status": rec.status, "run_id": rec.run_id}
    if resumed is not None:
        out["run"] = resumed
    if resume_error:
        out["resume_error"] = resume_error
    return out


@router.post("/approvals/{approval_id}/deny")
def deny(
    approval_id: str,
    body: ApprovalDecision | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    store = get_approval_store()
    rec = store.deny(approval_id, reason=(body.reason if body else None))
    if rec is None:
        raise HTTPException(404, "approval not found")
    run_id = _coerce_run_id(rec.run_id)
    if run_id is not None:
        row = db.get(AgentRun, run_id)
        if row is not None and row.state == "awaiting_approval":
            row.state = "cancelled"
            row.result = {
                **(row.result or {}),
                "checkpoint": None,
                "stopped": rec.reason or "denied",
            }
            db.commit()
    return {"id": rec.id, "status": rec.status, "run_id": rec.run_id}
