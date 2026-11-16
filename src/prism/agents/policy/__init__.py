"""ActionGate: fail-closed policy for tool execution."""

from prism.agents.policy.approvals import ApprovalStore, get_approval_store
from prism.agents.policy.audit import AuditLog, redact_args
from prism.agents.policy.gate import ActionGate, GateDecision, GateResult
from prism.agents.policy.rules import (
    PolicyBundle,
    PolicyRule,
    apply_overlay,
    load_policy,
    merge_rules,
)

__all__ = [
    "ActionGate",
    "ApprovalStore",
    "AuditLog",
    "GateDecision",
    "GateResult",
    "PolicyBundle",
    "PolicyRule",
    "apply_overlay",
    "get_approval_store",
    "load_policy",
    "merge_rules",
    "redact_args",
]
