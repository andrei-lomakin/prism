"""AgentRuntime: plan → act → synthesize, state machine, limits, subagents."""

from __future__ import annotations

from enum import StrEnum

from prism.agents.runtime.bus import MessageBus, get_bus
from prism.agents.runtime.limits import LimitTracker, RunBudget, RunUsage
from prism.agents.runtime.runtime import AgentAnswer, AgentRuntime
from prism.agents.runtime.subagents import SubagentBatch, SubagentManager
from prism.agents.runtime.triage import TriageResult, triage


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


__all__ = [
    "AgentAnswer",
    "AgentRuntime",
    "LimitTracker",
    "MessageBus",
    "RunBudget",
    "RunState",
    "RunUsage",
    "SubagentBatch",
    "SubagentManager",
    "TriageResult",
    "get_bus",
    "triage",
]
