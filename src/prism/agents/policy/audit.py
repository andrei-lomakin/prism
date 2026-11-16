"""ActionGate audit trail with secret redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_SECRET_KEYS = re.compile(
    r"(api[_-]?key|token|password|secret|authorization|bearer)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)


def redact_value(key: str, value: Any) -> Any:
    if _SECRET_KEYS.search(key):
        return "***"
    if isinstance(value, str):
        return _BEARER.sub(r"\1***", value)
    if isinstance(value, dict):
        return {k: redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(key, v) for v in value]
    return value


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: redact_value(k, v) for k, v in args.items()}


@dataclass
class AuditEvent:
    ts: str
    run_id: str | int | None
    tool: str
    rule_id: str
    mode: str
    decision: str  # allow | deny | ask | handoff | error
    args: dict[str, Any] = field(default_factory=dict)
    result_summary: str | None = None
    error: str | None = None


class AuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(
        self,
        *,
        run_id: str | int | None,
        tool: str,
        rule_id: str,
        mode: str,
        decision: str,
        args: dict[str, Any],
        result_summary: str | None = None,
        error: str | None = None,
    ) -> AuditEvent:
        ev = AuditEvent(
            ts=datetime.now(UTC).isoformat(),
            run_id=run_id,
            tool=tool,
            rule_id=rule_id,
            mode=mode,
            decision=decision,
            args=redact_args(args),
            result_summary=result_summary,
            error=error,
        )
        self.events.append(ev)
        return ev
