"""In-process subagent batch: parent parks, children run, aggregate returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4


@dataclass
class ChildRun:
    id: int
    role: str
    objective: str
    state: str = "queued"
    result: dict[str, Any] | None = None


@dataclass
class SubagentBatch:
    batch_id: str
    parent_id: str | int | None
    children: list[ChildRun] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return all(c.state in ("completed", "failed", "cancelled") for c in self.children)

    def aggregate(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "done": self.done,
            "children": [
                {
                    "id": c.id,
                    "role": c.role,
                    "state": c.state,
                    "result": c.result,
                }
                for c in self.children
            ],
        }


class SubagentManager:
    """Sync MVP: children execute via callback immediately (no Redis required)."""

    def __init__(self) -> None:
        self._batches: dict[str, SubagentBatch] = {}
        self._by_child: dict[int, str] = {}
        self._seq = 0

    def spawn(
        self,
        *,
        parent_id: str | int | None,
        roles: list[str],
        objective: str,
        runner: Callable[[ChildRun], dict[str, Any]],
    ) -> dict[str, Any]:
        batch_id = uuid4().hex[:12]
        children: list[ChildRun] = []
        for role in roles:
            self._seq += 1
            child = ChildRun(id=self._seq, role=role, objective=objective, state="running")
            children.append(child)
            self._by_child[child.id] = batch_id

        batch = SubagentBatch(batch_id=batch_id, parent_id=parent_id, children=children)
        self._batches[batch_id] = batch

        for child in children:
            try:
                child.result = runner(child)
                child.state = "completed"
            except Exception as exc:  # noqa: BLE001
                child.result = {"error": str(exc)}
                child.state = "failed"

        return batch.aggregate()

    def status(self, batch_id: str | None = None) -> dict[str, Any]:
        if batch_id is None:
            return {
                "batches": [b.aggregate() for b in self._batches.values()],
            }
        batch = self._batches.get(batch_id)
        if batch is None:
            return {"error": "not_found", "batch_id": batch_id}
        return batch.aggregate()

    def followup(self, child_id: int, message: str) -> dict[str, Any]:
        batch_id = self._by_child.get(child_id)
        if not batch_id:
            return {"error": "not_found", "child_id": child_id}
        return {
            "child_id": child_id,
            "accepted": False,
            "reason": "followup not supported in sync MVP",
            "message": message,
        }

    def stop(self, child_id: int) -> dict[str, Any]:
        batch_id = self._by_child.get(child_id)
        if not batch_id:
            return {"error": "not_found", "child_id": child_id}
        batch = self._batches[batch_id]
        for child in batch.children:
            if child.id == child_id and child.state not in ("completed", "failed"):
                child.state = "cancelled"
                child.result = {"stopped": True}
        return {"child_id": child_id, "stopped": True}
