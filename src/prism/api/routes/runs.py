"""Agent runs, traces, and approval queue."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from prism.agents.policy import get_approval_store
from prism.db.models import AgentRun, Step
from prism.db.session import get_db
from prism.observability import trace_buffer

router = APIRouter(prefix="/api", tags=["runs"])


class ApprovalDecision(BaseModel):
    reason: str | None = None


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


@router.post("/approvals/{approval_id}/approve")
def approve(approval_id: str, body: ApprovalDecision | None = None) -> dict[str, Any]:
    store = get_approval_store()
    rec = store.approve(approval_id, reason=(body.reason if body else None))
    if rec is None:
        raise HTTPException(404, "approval not found")
    return {"id": rec.id, "status": rec.status}


@router.post("/approvals/{approval_id}/deny")
def deny(approval_id: str, body: ApprovalDecision | None = None) -> dict[str, Any]:
    store = get_approval_store()
    rec = store.deny(approval_id, reason=(body.reason if body else None))
    if rec is None:
        raise HTTPException(404, "approval not found")
    return {"id": rec.id, "status": rec.status}
