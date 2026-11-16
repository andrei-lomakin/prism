"""AgentRuntime: plan → act → synthesize, state machine, limits, subagents."""

from __future__ import annotations

from enum import StrEnum


class RunState(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_SUBAGENTS = "awaiting_subagents"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class AgentRuntime:
    """Orchestrates a single AgentRun. Implemented in stage 4."""

    def run(self, run_id: int) -> None:
        raise NotImplementedError("AgentRuntime lands in stage 4")
