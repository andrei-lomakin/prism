"""ActionGate: sole path to tool.execute (fail-closed)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prism.agents.policy.approvals import ApprovalStore, get_approval_store
from prism.agents.policy.audit import AuditLog
from prism.agents.policy.rules import Mode, PolicyBundle, load_policy
from prism.agents.tools import ToolSpec
from prism.observability import span

_BLOB_PARAM_NAMES = frozenset({"args", "payload", "params"})

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
    approval_id: str | None = None


ApprovalFn = Callable[[GateDecision, ToolSpec, dict[str, Any]], bool]


class ActionGate:
    """Evaluate policy then optionally execute. Only this class calls tool.execute."""

    def __init__(
        self,
        bundle: PolicyBundle | None = None,
        *,
        audit: AuditLog | None = None,
        approval_fn: ApprovalFn | None = None,
        approvals: ApprovalStore | None = None,
    ) -> None:
        self.bundle = bundle or load_policy()
        self.audit = audit or AuditLog()
        self.approval_fn = approval_fn
        self.approvals = approvals if approvals is not None else get_approval_store()

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

        with span(
            "agent.gate",
            attributes={"tool": tool.code, "mode": decision.mode, "rule_id": decision.rule_id},
        ):
            return self._execute_inner(run_id=run_id, tool=tool, args=args, decision=decision)

    def _execute_inner(
        self,
        *,
        run_id: str | int | None,
        tool: ToolSpec,
        args: dict[str, Any],
        decision: GateDecision,
    ) -> GateResult:
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
            approved = self.approvals.is_approved(run_id=run_id, tool=tool.code, args=args)
            if not approved and self.approval_fn is not None:
                approved = bool(self.approval_fn(decision, tool, args))
            if not approved:
                rec = self.approvals.request(
                    run_id=run_id,
                    tool=tool.code,
                    args=args,
                    rule_id=decision.rule_id,
                )
                self.audit.record(
                    run_id=run_id,
                    tool=tool.code,
                    rule_id=decision.rule_id,
                    mode=decision.mode,
                    decision="ask",
                    args=args,
                )
                return GateResult(
                    decision=decision,
                    executed=False,
                    error="approval required",
                    approval_id=rec.id,
                )

        try:
            output = invoke_tool(tool.execute, args)
        except Exception as exc:  # noqa: BLE001 — audited; never retry
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


def invoke_tool(execute: Callable[..., Any], args: dict[str, Any]) -> Any:
    """Bind args to the tool signature *before* calling. No exception-driven retry."""
    sig = inspect.signature(execute)
    params = list(sig.parameters.values())

    # Legacy: single blob param `def tool(args|payload|params): ...`
    if (
        len(params) == 1
        and params[0].kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and params[0].name in _BLOB_PARAM_NAMES
        and params[0].name not in args
    ):
        return execute(args)

    try:
        bound = sig.bind(**args) if args else sig.bind()
    except TypeError as exc:
        raise TypeError(f"tool args mismatch for {execute!r}: {exc}") from exc
    bound.apply_defaults()
    return execute(*bound.args, **bound.kwargs)


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
