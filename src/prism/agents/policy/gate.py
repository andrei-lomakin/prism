"""ActionGate: sole path to tool.execute (fail-closed)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prism.agents.policy.audit import AuditLog
from prism.agents.policy.rules import Mode, PolicyBundle, load_policy
from prism.agents.tools import ToolSpec

DecisionMode = Mode


@dataclass
class GateDecision:
    mode: DecisionMode
    rule_id: str
    reason: str


@dataclass
class GateResult:
    decision: GateDecision
    executed: bool
    output: Any = None
    error: str | None = None


ApprovalFn = Callable[[GateDecision, ToolSpec, dict[str, Any]], bool]


class ActionGate:
    """Evaluate policy then optionally execute. Only this class calls tool.execute."""

    def __init__(
        self,
        bundle: PolicyBundle | None = None,
        *,
        audit: AuditLog | None = None,
        approval_fn: ApprovalFn | None = None,
    ) -> None:
        self.bundle = bundle or load_policy()
        self.audit = audit or AuditLog()
        self.approval_fn = approval_fn

    def evaluate(self, run: Any, tool: Any, args: dict) -> GateDecision:
        code = getattr(tool, "code", None) or getattr(tool, "__name__", None) or "unknown"
        rule = self.bundle.rule_for(str(code))
        if rule is None:
            return GateDecision(
                mode=self.bundle.default_mode,
                rule_id="default.unknown",
                reason="no matching rule",
            )
        return GateDecision(mode=rule.mode, rule_id=rule.id, reason=f"matched {rule.layer}")

    def execute(
        self,
        *,
        run: Any,
        tool: ToolSpec,
        args: dict[str, Any],
    ) -> GateResult:
        """Sole entry that may invoke tool.execute."""
        run_id = getattr(run, "id", None) if run is not None else None
        decision = self.evaluate(run, tool, args)

        if decision.mode == "handoff":
            self.audit.record(
                run_id=run_id,
                tool=tool.code,
                rule_id=decision.rule_id,
                mode=decision.mode,
                decision="handoff",
                args=args,
            )
            return GateResult(decision=decision, executed=False, error="handoff required")

        if decision.mode in ("ask", "on_request"):
            approved = False
            if self.approval_fn is not None:
                approved = bool(self.approval_fn(decision, tool, args))
            if not approved:
                self.audit.record(
                    run_id=run_id,
                    tool=tool.code,
                    rule_id=decision.rule_id,
                    mode=decision.mode,
                    decision="ask",
                    args=args,
                )
                return GateResult(decision=decision, executed=False, error="approval required")

        try:
            output = tool.execute(**args) if args else tool.execute()
        except TypeError:
            # Tools that take a single dict payload
            output = tool.execute(args)
        except Exception as exc:  # noqa: BLE001 — audited
            self.audit.record(
                run_id=run_id,
                tool=tool.code,
                rule_id=decision.rule_id,
                mode=decision.mode,
                decision="error",
                args=args,
                error=str(exc),
            )
            return GateResult(decision=decision, executed=False, error=str(exc))

        summary = _summarize(output)
        self.audit.record(
            run_id=run_id,
            tool=tool.code,
            rule_id=decision.rule_id,
            mode=decision.mode,
            decision="allow",
            args=args,
            result_summary=summary,
        )
        return GateResult(decision=decision, executed=True, output=output)


def _summarize(output: Any) -> str:
    if output is None:
        return "null"
    if isinstance(output, dict):
        keys = ",".join(sorted(output.keys())[:8])
        return f"dict[{keys}]"
    if isinstance(output, list):
        return f"list(n={len(output)})"
    text = str(output)
    return text if len(text) <= 120 else text[:117] + "..."
