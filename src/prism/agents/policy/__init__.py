"""ActionGate: fail-closed policy for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Mode = Literal["auto", "on_request", "ask", "handoff"]


@dataclass
class GateDecision:
    mode: Mode
    rule_id: str
    reason: str


class ActionGate:
    """Sole path to tool.execute (enforced by tests in stage 4)."""

    def evaluate(self, run: Any, tool: Any, args: dict) -> GateDecision:
        # Unknown → ask (fail closed)
        return GateDecision(mode="ask", rule_id="default.unknown", reason="no matching rule")
