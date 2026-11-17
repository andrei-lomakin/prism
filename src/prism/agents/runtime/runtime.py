"""AgentRuntime: triage → plan → act (via ActionGate) → synthesize."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from prism.agents.memory import MemoryStore
from prism.agents.policy import ActionGate, ApprovalStore, PolicyBundle, load_policy
from prism.agents.policy.audit import AuditLog
from prism.agents.roles import run_analyst, run_critic, run_retriever, run_web
from prism.agents.runtime.bus import get_bus
from prism.agents.runtime.limits import LimitTracker, RunBudget, fingerprint_call, usage_from_dict
from prism.agents.runtime.subagents import ChildRun, SubagentManager
from prism.agents.runtime.triage import TriageResult, triage
from prism.agents.tools import ToolContext, ToolRegistry, build_default_registry
from prism.agents.web_evidence import INSUFFICIENT_NOTE, evidence_snippets, user_web_note
from prism.config import Settings, get_settings
from prism.db.models import AgentRun, Step
from prism.llm import LLMGateway
from prism.llm.gateway import LLMResult
from prism.net.safe_http import UnsafeUrlError, assert_public_http_url
from prism.rag.generate import SYSTEM_PROMPT, build_rag_prompt, check_grounding


class RunState:
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_SUBAGENTS = "awaiting_subagents"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class AgentAnswer:
    run_id: str
    answer: str
    model: str
    citations: list[dict[str, Any]]
    retrieval_mode: str
    grounding_ok: bool
    usage: dict[str, Any]
    state: str
    partial: bool = False
    plan: dict[str, Any] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    approval: dict[str, Any] | None = None


class _EphemeralRun:
    """Stand-in when DB is unavailable (unit tests)."""

    def __init__(self, role: str, objective: str) -> None:
        self.id: str | int = uuid4().hex[:10]
        self.role = role
        self.objective = objective
        self.state = RunState.QUEUED
        self.plan: dict[str, Any] | None = None
        self.budget: dict[str, Any] = {}
        self.usage: dict[str, Any] = {}
        self.result: dict[str, Any] | None = None
        self.parent_id: int | None = None


_EPHEMERAL_RUNS: dict[str, _EphemeralRun] = {}
_SETTING_KEYS = (
    "space_filter",
    "web_search_mock",
    "top_k",
    "retrieval_mode",
    "embedding_model",
    "rewrite_strategy",
    "rerank_enabled",
    "hybrid_fusion",
    "distill_enabled",
)


class AgentRuntime:
    """Orchestrates one AgentRun (sync MVP; Redis queue later)."""

    def __init__(
        self,
        db: Session | None = None,
        *,
        settings: Settings | None = None,
        llm: LLMGateway | Callable[..., LLMResult] | None = None,
        budget: RunBudget | None = None,
        policy: PolicyBundle | None = None,
        approval_fn: Callable[..., bool] | None = None,
        approvals: ApprovalStore | None = None,
        persist: bool = True,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.budget = budget or RunBudget()
        self.audit = AuditLog()
        self.policy = policy or load_policy()
        self.gate = ActionGate(
            self.policy, audit=self.audit, approval_fn=approval_fn, approvals=approvals
        )
        self.memory = MemoryStore(db)
        self.subagents = SubagentManager()
        self.limits = LimitTracker(self.budget)
        self.persist = persist and db is not None
        self._llm = llm
        self._on_step = on_step
        self._on_status = on_status
        self._steps: list[dict[str, Any]] = []
        self._step_n = 0
        self._resuming = False

        self.ctx = ToolContext(
            db=db,
            settings=self.settings,
            memory=self.memory,
            spawn_fn=self._tool_spawn,
            status_fn=self._tool_status,
            followup_fn=self._tool_followup,
            stop_fn=self._tool_stop,
            get_policy=lambda: self.policy,
            set_policy=self._set_policy,
        )
        self.registry: ToolRegistry = build_default_registry(self.ctx)
        self._active_run: Any = None
        self._batch_id: str | None = None

    def run(self, run_id: int) -> None:
        if self.db is None:
            raise NotImplementedError("DB run_id path requires a session")
        row = self.db.get(AgentRun, run_id)
        if row is None:
            raise ValueError(f"AgentRun {run_id} not found")
        answer = self._execute(row, row.objective, role=row.role)
        row.state = answer.state
        row.result = {
            "answer": answer.answer,
            "partial": answer.partial,
            "grounding_ok": answer.grounding_ok,
        }
        row.usage = answer.usage
        row.plan = answer.plan
        self.db.commit()

    def handle(self, objective: str, *, role: str = "orchestrator") -> AgentAnswer:
        run = self._create_run(role=role, objective=objective)
        return self._execute(run, objective, role=role)

    def resume(self, run_id: str | int | None = None) -> AgentAnswer:
        """Continue the same run. Restores checkpoint context, budget, and phase."""
        if self._resuming:
            raise RuntimeError("resume already in progress")
        run = self._run_for_resume(run_id)
        result = dict(getattr(run, "result", None) or {})
        stage = result.get("failure_stage")
        state = getattr(run, "state", None)
        if state == RunState.FAILED and stage == "synthesize":
            checkpoint = dict(result.get("checkpoint") or {})
            self._restore_checkpoint(run, checkpoint)
            t0 = perf_counter()
            self._active_run = run
            self._resuming = True
            try:
                return self._finish_answer(run, checkpoint, t0, resumed=True)
            finally:
                self._resuming = False
        if state != RunState.AWAITING_APPROVAL:
            raise RuntimeError(
                f"run {getattr(run, 'id', run_id)} is not awaiting_approval (state={state})"
            )
        checkpoint = dict(result.get("checkpoint") or {})
        tool_code, args = self._pending_action(checkpoint)
        if not tool_code:
            raise RuntimeError("run has no approval checkpoint")
        fp = fingerprint_call(tool_code, args)
        completed = list(checkpoint.get("completed") or [])
        if fp not in completed and not self.gate.approvals.is_approved(
            run_id=getattr(run, "id", None), tool=tool_code, args=args
        ):
            raise RuntimeError(
                f"approval for {tool_code} is not active; approve the original action first"
            )
        self._restore_checkpoint(run, checkpoint)
        t0 = perf_counter()
        self._active_run = run
        self._resuming = True
        run.state = RunState.RUNNING
        self._commit()
        try:
            if fp in completed:
                output = dict(checkpoint.get("last_output") or {})
            else:
                tool = self.registry.get(tool_code)
                if tool is None:
                    return self._fail_kept(
                        run,
                        checkpoint,
                        f"unknown tool {tool_code}",
                        stage="tool",
                        tool=tool_code,
                        args=args,
                        t0=t0,
                    )
                gr = self.gate.execute(run=run, tool=tool, args=args)
                self._record_step(
                    "tool",
                    {"tool": tool_code, "resumed": True, "args": args},
                    gr.output if gr.executed else {"error": gr.error, "outcome": gr.outcome},
                )
                if gr.outcome == "approval_required" and gr.approval_id:
                    if self.gate.approvals.get(gr.approval_id) is not None:
                        return self._pause_for_approval(run, gr.approval_id, t0, checkpoint)
                if not gr.executed or gr.outcome == "error":
                    checkpoint["attempted"] = {
                        "tool": tool_code,
                        "args": args,
                        "error": gr.error,
                    }
                    return self._fail_kept(
                        run,
                        checkpoint,
                        gr.error or "approved action failed",
                        stage="tool",
                        tool=tool_code,
                        args=args,
                        t0=t0,
                    )
                output = gr.output if isinstance(gr.output, dict) else {"value": gr.output}
                completed.append(fp)
                checkpoint["completed"] = completed
                checkpoint["last_output"] = output
                checkpoint["pending"] = None
                self._consume_tool_output(checkpoint, tool_code, output)
                run.result = {"checkpoint": checkpoint}
                run.usage = self.limits.usage.to_dict()
                self._commit()
            return self._continue_workflow(run, checkpoint, t0)
        finally:
            self._resuming = False

    # --- internals ---

    def _set_policy(self, bundle: PolicyBundle) -> None:
        self.policy = bundle
        self.gate.bundle = bundle

    def _create_run(self, *, role: str, objective: str) -> Any:
        if self.persist and self.db is not None:
            row = AgentRun(
                role=role,
                objective=objective,
                state=RunState.QUEUED,
                budget=self._budget_dict(),
                usage={},
            )
            self.db.add(row)
            self.db.flush()
            return row
        run = _EphemeralRun(role=role, objective=objective)
        run.budget = self._budget_dict()
        _EPHEMERAL_RUNS[str(run.id)] = run
        return run

    def _execute(self, run: Any, objective: str, *, role: str) -> AgentAnswer:
        t0 = perf_counter()
        self._active_run = run
        self.limits = LimitTracker(self.budget)
        self._steps = []
        self._step_n = 0
        run.state = RunState.PLANNING
        self._commit()

        tri: TriageResult = triage(objective)
        plan = {
            "complexity": tri.complexity,
            "roles": tri.roles,
            "reason": tri.reason,
            "steps": tri.plan_steps,
        }
        run.plan = plan
        run.state = RunState.RUNNING
        self._record_step("plan", {"objective": objective}, plan)
        self._commit()

        hits: list[dict[str, Any]] = []
        retrieval_mode = "none"
        critic_report: dict[str, Any] | None = None
        analyst_report: dict[str, Any] | None = None
        web_report: dict[str, Any] | None = None
        awaiting = False
        approval_id: str | None = None

        try:
            if role == "retriever":
                out = run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=run,
                )
                hits = list(out.get("hits") or [])
                retrieval_mode = str(out.get("retrieval_mode") or "retriever")
                self._record_step("tool", {"tool": "search_kb"}, out)
            elif role == "web":
                web_report = run_web(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=run,
                )
                self._record_step("tool", {"tool": "web_search"}, web_report)
                if web_report.get("error") == "approval required":
                    awaiting = True
                    approval_id = web_report.get("approval_id")
            elif role == "analyst":
                analyst_report = run_analyst(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=run,
                )
                self._record_step("tool", {"tool": "build_comparison_table"}, analyst_report)
            elif role == "critic":
                ret = run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=run,
                )
                hits = list(ret.get("hits") or [])
                web = list((web_report or {}).get("results") or [])
                _, source_texts, _ = self._merge_sources(hits, web)
                draft = self._draft_from_hits(objective, hits, web)
                critic_report = run_critic(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    answer=draft,
                    sources=source_texts,
                    run=run,
                )
                self._record_step("tool", {"tool": "verify_citations"}, critic_report)
            else:
                # Orchestrator
                if tri.complexity in ("multi", "research"):
                    run.state = RunState.AWAITING_SUBAGENTS
                    self._commit()
                    get_bus().post(
                        f"run:{run.id}",
                        f"spawn {tri.roles}",
                        sender="orchestrator",
                        urgent=True,
                    )
                    spawn_tool = self.registry.get("spawn_subagents")
                    assert spawn_tool is not None
                    self.limits.note_step()
                    roles_s = ", ".join(tri.roles)
                    self._emit_status(f"Делегирование субагентам ({roles_s})…")
                    self._record_step(
                        "delegation",
                        {"roles": tri.roles, "phase": "start"},
                        {"roles": tri.roles},
                    )
                    gr = self.gate.execute(
                        run=run,
                        tool=spawn_tool,
                        args={"roles": tri.roles, "objective": objective},
                    )
                    self._record_step(
                        "delegation",
                        {"roles": tri.roles, "phase": "done"},
                        gr.output,
                    )
                    if not gr.executed:
                        if gr.decision.mode in ("ask", "on_request"):
                            run.state = RunState.AWAITING_APPROVAL
                            awaiting = True
                            approval_id = gr.approval_id
                        else:
                            self.limits.mark_forced()
                    else:
                        agg = gr.output or {}
                        (
                            hits,
                            retrieval_mode,
                            critic_report,
                            analyst_report,
                            web_report,
                        ) = self._harvest_batch(agg)
                        if web_report and web_report.get("error") == "approval required":
                            run.state = RunState.AWAITING_APPROVAL
                            awaiting = True
                            approval_id = web_report.get("approval_id")
                        else:
                            run.state = RunState.RUNNING
                else:
                    if self.limits.would_exceed():
                        self.limits.mark_forced()
                    else:
                        out = run_retriever(
                            gate=self.gate,
                            registry=self.registry,
                            limits=self.limits,
                            objective=objective,
                            run=run,
                        )
                        hits = list(out.get("hits") or [])
                        retrieval_mode = str(out.get("retrieval_mode") or "agent")
                        self._record_step("tool", {"tool": "search_kb"}, out)
                        if not hits and out.get("error") == "approval required":
                            run.state = RunState.AWAITING_APPROVAL
                            awaiting = True
                            approval_id = out.get("approval_id")

            workflow = self._workflow(
                objective=objective,
                role=role,
                plan=plan,
                hits=hits,
                retrieval_mode=retrieval_mode,
                web_report=web_report,
                analyst_report=analyst_report,
                critic_report=critic_report,
            )
            if awaiting:
                return self._pause_for_approval(run, approval_id, t0, workflow)

            if self.limits.would_exceed():
                self.limits.mark_forced()

            return self._finish_answer(run, workflow, t0, resumed=False)
        except Exception as exc:  # noqa: BLE001
            prev = run.result if isinstance(getattr(run, "result", None), dict) else {}
            run.state = RunState.FAILED
            run.result = {
                "error": str(exc),
                "failure_stage": prev.get("failure_stage") or "runtime",
                "checkpoint": prev.get("checkpoint") if prev.get("checkpoint") is not None else {},
            }
            self._commit()
            raise

    def _run_for_resume(self, run_id: str | int | None) -> Any:
        if self.db is not None and run_id is not None:
            try:
                key = int(run_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"run id {run_id} is not a persisted run") from exc
            row = self.db.get(AgentRun, key)
            if row is None:
                raise RuntimeError(f"run {run_id} not found")
            return row
        if run_id is not None and str(run_id) in _EPHEMERAL_RUNS:
            return _EPHEMERAL_RUNS[str(run_id)]
        run = self._active_run
        if run is None:
            raise RuntimeError("no active run to resume")
        if run_id is not None and str(run.id) != str(run_id):
            raise RuntimeError(f"run {run_id} is not the active run {run.id}")
        return run

    def _sync_step_counter(self, run: Any) -> None:
        if self.persist and self.db is not None and isinstance(getattr(run, "id", None), int):
            n = self.db.query(func.max(Step.n)).filter(Step.run_id == run.id).scalar()
            self._step_n = int(n or 0)
            return
        self._step_n = len(self._steps)

    def _budget_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.budget.max_steps,
            "max_searches": self.budget.max_searches,
            "max_tokens": self.budget.max_tokens,
            "max_cost": self.budget.max_cost_usd,
            "max_cost_usd": self.budget.max_cost_usd,
            "max_active_seconds": self.budget.max_active_seconds,
        }

    def _settings_snapshot(self) -> dict[str, Any]:
        data = self.settings.model_dump()
        return {key: data[key] for key in _SETTING_KEYS if key in data}

    def _note_slice(self, t0: float) -> None:
        self.limits.note_active(perf_counter() - t0)

    def _restore_checkpoint(self, run: Any, cp: dict[str, Any]) -> None:
        saved = {
            key: value
            for key, value in dict(cp.get("settings") or {}).items()
            if key in _SETTING_KEYS
        }
        if saved:
            self.settings = self.settings.model_copy(update=saved)
            self.ctx.settings = self.settings
        budget = dict(cp.get("budget") or {})
        if budget:
            self.budget = RunBudget(
                max_steps=int(budget.get("max_steps") or self.budget.max_steps),
                max_searches=int(budget.get("max_searches") or self.budget.max_searches),
                max_tokens=int(budget.get("max_tokens") or self.budget.max_tokens),
                max_cost_usd=float(
                    budget.get("max_cost_usd", budget.get("max_cost", self.budget.max_cost_usd))
                ),
                max_active_seconds=float(
                    budget.get("max_active_seconds", self.budget.max_active_seconds)
                ),
            )
        self.limits = LimitTracker(self.budget)
        self.limits.usage = usage_from_dict(cp.get("usage") or {})
        self._steps = [dict(step) for step in (cp.get("steps") or [])]
        self._step_n = int(cp.get("step_n") or len(self._steps))
        if self.persist and self.db is not None and isinstance(getattr(run, "id", None), int):
            n = self.db.query(func.max(Step.n)).filter(Step.run_id == run.id).scalar()
            self._step_n = max(self._step_n, int(n or 0))

    def _workflow(
        self,
        *,
        objective: str,
        role: str,
        plan: dict[str, Any] | None,
        hits: list[dict[str, Any]],
        retrieval_mode: str,
        web_report: dict[str, Any] | None,
        analyst_report: dict[str, Any] | None,
        critic_report: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del critic_report  # final critic runs after synthesis, not on the spawn draft
        web = web_report or {}
        return {
            "phase": web.get("phase") or "tool",
            "role": role,
            "roles": list((plan or {}).get("roles") or []),
            "complexity": (plan or {}).get("complexity"),
            "objective": objective,
            "hits": list(hits or []),
            "retrieval_mode": retrieval_mode,
            "web_results": list(web.get("results") or []),
            "pages": list(web.get("pages") or []),
            "fetch_queue": list(web.get("fetch_queue") or []),
            "web_unavailable": bool(web.get("web_unavailable")),
            "web_status": web.get("status"),
            "web_origin": web.get("origin"),
            "web_error": web.get("error") if web.get("web_unavailable") else None,
            "analyst": analyst_report,
            "critic_done": False,
            "plan": plan,
            "settings": self._settings_snapshot(),
            "space": self.settings.space_filter,
            "completed": [],
            "step_n": self._step_n,
            "steps": list(self._steps),
            "budget": self._budget_dict(),
            "usage": self.limits.usage.to_dict(),
        }

    @staticmethod
    def _pending_action(cp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        pending = cp.get("pending") if isinstance(cp.get("pending"), dict) else {}
        tool = str(pending.get("tool") or cp.get("tool") or "")
        args = dict(pending.get("args") or cp.get("args") or {})
        return tool, args

    def _stamp_checkpoint(self, cp: dict[str, Any]) -> dict[str, Any]:
        stamped = dict(cp)
        stamped["step_n"] = self._step_n
        stamped["steps"] = list(self._steps)
        stamped["usage"] = self.limits.usage.to_dict()
        stamped["budget"] = self._budget_dict()
        stamped["settings"] = stamped.get("settings") or self._settings_snapshot()
        return stamped

    def _pause_for_approval(
        self,
        run: Any,
        approval_id: str | None,
        t0: float,
        workflow: dict[str, Any] | None = None,
    ) -> AgentAnswer:
        rec = self.gate.approvals.get(approval_id) if approval_id else None
        if rec is None:
            return self._fail_kept(
                run,
                workflow,
                "Одобрение не найдено, действие не выполнялось.",
                stage="approval",
                t0=t0,
            )
        self._note_slice(t0)
        cp = self._stamp_checkpoint(workflow or {})
        cp["pending"] = {
            "tool": rec.tool,
            "args": dict(rec.args),
            "approval_id": rec.id,
            "rule_id": rec.rule_id,
        }
        cp["tool"] = rec.tool
        cp["args"] = dict(rec.args)
        cp["approval_id"] = rec.id
        run.state = RunState.AWAITING_APPROVAL
        run.usage = self.limits.usage.to_dict()
        run.result = {
            "checkpoint": cp,
            "approval_id": rec.id,
            "retrieval_mode": cp.get("retrieval_mode"),
            "plan": cp.get("plan"),
        }
        self._commit()
        latency = int((perf_counter() - t0) * 1000)
        return AgentAnswer(
            run_id=str(run.id),
            answer=f"Требуется одобрение действия (ActionGate ask). approval_id={rec.id}",
            model="none",
            citations=[],
            retrieval_mode=str(cp.get("retrieval_mode") or "none"),
            grounding_ok=False,
            usage=self.limits.usage.to_dict(),
            state=RunState.AWAITING_APPROVAL,
            plan=dict(cp.get("plan") or run.plan or {}),
            audit=self._audit_dicts(),
            steps=list(self._steps),
            latency_ms=latency,
            approval={
                "id": rec.id,
                "run_id": str(run.id),
                "tool": rec.tool,
                "args": dict(rec.args),
            },
        )

    def _fail_kept(
        self,
        run: Any,
        workflow: dict[str, Any] | None,
        message: str,
        *,
        stage: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        t0: float | None = None,
    ) -> AgentAnswer:
        if t0 is not None:
            self._note_slice(t0)
        cp = self._stamp_checkpoint(workflow or {})
        if tool is not None:
            cp["attempted"] = {"tool": tool, "args": args, "error": message}
        run.state = RunState.FAILED
        run.usage = self.limits.usage.to_dict()
        run.result = {
            "error": message,
            "failure_stage": stage,
            "tool": tool,
            "args": args,
            "checkpoint": cp,
        }
        self._commit()
        if stage == "tool":
            text = f"Одобрено, но выполнение завершилось ошибкой: {message}"
        elif stage == "synthesize":
            text = f"Ошибка синтеза после успешного действия: {message}"
        else:
            text = message
        return AgentAnswer(
            run_id=str(run.id),
            answer=text,
            model="none",
            citations=[],
            retrieval_mode=str(cp.get("retrieval_mode") or "none"),
            grounding_ok=False,
            usage=self.limits.usage.to_dict(),
            state=RunState.FAILED,
            plan=dict(cp.get("plan") or getattr(run, "plan", None) or {}),
            audit=self._audit_dicts(),
            steps=list(self._steps),
            latency_ms=0,
        )

    def _absorb_search(self, cp: dict[str, Any], output: dict[str, Any]) -> None:
        mock_mode = bool(self.settings.web_search_mock)
        status = str(output.get("status") or ("ok" if output.get("results") else "empty"))
        cp["web_status"] = status
        cp["web_origin"] = output.get("origin")
        cp["web_error"] = output.get("error")
        kept = evidence_snippets(list(output.get("results") or []), None, mock_mode=mock_mode)
        safe_rows: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for item in kept:
            url = str(item.get("url") or "")
            if url and not self._url_allowed(url):
                blocked.append({"url": url, "error": "blocked address", "origin": "blocked"})
                continue
            safe_rows.append(item)
        if blocked:
            cp["page_errors"] = list(cp.get("page_errors") or []) + blocked
        cp["web_results"] = safe_rows
        kept = safe_rows
        blocked_all = bool(blocked) and not safe_rows
        if status in ("error", "empty") or output.get("ok") is False or blocked_all:
            cp["web_unavailable"] = True
            if blocked_all and status == "ok":
                cp["web_status"] = "error"
                cp["web_error"] = "blocked address"
            cp["fetch_queue"] = []
            cp["phase"] = "web_done"
            return
        urls: list[str] = []
        for item in kept:
            url = str(item.get("url") or "")
            if url and url not in urls:
                urls.append(url)
            if len(urls) == 2:
                break
        done = {str(page.get("url") or "") for page in cp.get("pages") or []}
        cp["fetch_queue"] = [url for url in urls if url not in done]
        cp["phase"] = "fetch_page" if cp["fetch_queue"] else "web_done"

    def _drop_unsafe(self, rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for item in rows or []:
            url = str(item.get("url") or "")
            if url and not self._url_allowed(url):
                continue
            kept.append(item)
        return kept

    def _url_allowed(self, url: str) -> bool:
        try:
            assert_public_http_url(url, resolve=not bool(self.settings.web_search_mock))
        except UnsafeUrlError:
            return False
        return True

    def _absorb_page(self, cp: dict[str, Any], output: dict[str, Any]) -> None:
        queue = list(cp.get("fetch_queue") or [])
        if queue:
            queue.pop(0)
        cp["fetch_queue"] = queue
        if output.get("ok") is False or output.get("origin") == "blocked":
            errors = list(cp.get("page_errors") or [])
            errors.append(
                {
                    "url": output.get("url"),
                    "error": output.get("error"),
                    "origin": output.get("origin"),
                }
            )
            cp["page_errors"] = errors
            return
        pages = list(cp.get("pages") or [])
        pages.append(output)
        cp["pages"] = pages

    def _consume_tool_output(self, cp: dict[str, Any], tool: str, output: dict[str, Any]) -> None:
        if tool == "web_search":
            self._absorb_search(cp, output)
        elif tool == "fetch_page":
            self._absorb_page(cp, output)
        elif tool == "search_kb":
            cp["hits"] = list(output.get("hits") or cp.get("hits") or [])
            if output.get("retrieval_mode"):
                cp["retrieval_mode"] = output["retrieval_mode"]

    def _continue_workflow(self, run: Any, cp: dict[str, Any], t0: float) -> AgentAnswer:
        paused = self._walk_fetches(run, cp, t0)
        if paused is not None:
            return paused
        return self._finish_answer(run, cp, t0, resumed=True)

    def _walk_fetches(self, run: Any, cp: dict[str, Any], t0: float) -> AgentAnswer | None:
        while cp.get("fetch_queue"):
            url = str(cp["fetch_queue"][0])
            args = {"url": url}
            fp = fingerprint_call("fetch_page", args)
            completed = list(cp.get("completed") or [])
            if fp in completed:
                cp["fetch_queue"] = list(cp["fetch_queue"])[1:]
                continue
            tool = self.registry.get("fetch_page")
            if tool is None:
                return self._fail_kept(
                    run,
                    cp,
                    "unknown tool fetch_page",
                    stage="tool",
                    tool="fetch_page",
                    args=args,
                    t0=t0,
                )
            self.limits.note_step()
            gr = self.gate.execute(run=run, tool=tool, args=args)
            self._record_step(
                "tool",
                {"tool": "fetch_page", "resumed": True, "args": args},
                gr.output if gr.executed else {"error": gr.error, "outcome": gr.outcome},
            )
            if gr.outcome == "approval_required" and gr.approval_id:
                if self.gate.approvals.get(gr.approval_id) is not None:
                    cp["phase"] = "fetch_page"
                    cp["pending"] = {
                        "tool": "fetch_page",
                        "args": args,
                        "approval_id": gr.approval_id,
                    }
                    return self._pause_for_approval(run, gr.approval_id, t0, cp)
            if not gr.executed or gr.outcome == "error":
                cp["attempted"] = {"tool": "fetch_page", "args": args, "error": gr.error}
                return self._fail_kept(
                    run,
                    cp,
                    gr.error or "fetch_page failed",
                    stage="tool",
                    tool="fetch_page",
                    args=args,
                    t0=t0,
                )
            output = gr.output if isinstance(gr.output, dict) else {"value": gr.output}
            completed.append(fp)
            cp["completed"] = completed
            cp["last_output"] = output
            cp["pending"] = None
            self._consume_tool_output(cp, "fetch_page", output)
            run.result = {"checkpoint": cp}
            run.usage = self.limits.usage.to_dict()
            self._commit()
        return None

    def _ensure_analyst(
        self, run: Any, workflow: dict[str, Any], snippets: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        roles = list(workflow.get("roles") or [])
        current = workflow.get("analyst")
        if "analyst" not in roles:
            return current if isinstance(current, dict) else None
        if isinstance(current, dict) and current.get("markdown") and not current.get("skipped"):
            return current
        if self.limits.would_exceed():
            return current if isinstance(current, dict) else None
        report = run_analyst(
            gate=self.gate,
            registry=self.registry,
            limits=self.limits,
            objective=str(workflow.get("objective") or getattr(run, "objective", "")),
            run=run,
            kb_hits=list(workflow.get("hits") or []),
            web_results=snippets,
        )
        workflow["analyst"] = report
        self._record_step("tool", {"tool": "build_comparison_table", "resumed": True}, report)
        return report

    def _finish_answer(
        self,
        run: Any,
        workflow: dict[str, Any],
        t0: float,
        *,
        resumed: bool,
    ) -> AgentAnswer:
        objective = str(workflow.get("objective") or getattr(run, "objective", "") or "")
        hits = list(workflow.get("hits") or [])
        plan = dict(workflow.get("plan") or getattr(run, "plan", None) or {})
        complexity = workflow.get("complexity") or plan.get("complexity")
        roles = list(workflow.get("roles") or plan.get("roles") or [])
        mock_mode = bool(self.settings.web_search_mock)
        web_report = {
            "results": list(workflow.get("web_results") or []),
            "pages": list(workflow.get("pages") or []),
            "status": workflow.get("web_status"),
            "origin": workflow.get("web_origin"),
            "web_unavailable": bool(workflow.get("web_unavailable")),
            "error": workflow.get("web_error"),
        }
        snippets = evidence_snippets(
            self._drop_unsafe(web_report["results"]),
            self._drop_unsafe(web_report["pages"]),
            mock_mode=mock_mode,
        )
        web_attempted = bool(
            workflow.get("web_status")
            or workflow.get("web_unavailable")
            or workflow.get("web_origin")
        )
        status = workflow.get("web_status")
        unavailable = bool(workflow.get("web_unavailable"))
        note = user_web_note(
            status=str(status) if status else None,
            unavailable=unavailable,
            snippets=snippets,
            has_kb=bool(hits),
            web_attempted=web_attempted,
        )
        analyst_report = self._ensure_analyst(run, workflow, snippets)
        table_md = (
            (analyst_report or {}).get("markdown") if isinstance(analyst_report, dict) else None
        )
        _, source_texts, citations = self._merge_sources(hits, snippets)
        workflow["saved_output"] = {
            "hits": hits,
            "web_results": snippets,
            "pages": list(workflow.get("pages") or []),
            "last_output": workflow.get("last_output"),
        }
        skip_llm = (
            not hits
            and not snippets
            and web_attempted
            and (unavailable or status in ("error", "empty"))
        )
        if self.limits.would_exceed():
            self.limits.mark_forced()
        model = "none"
        tin = tout = 0
        cost = 0.0
        llm_ms = 0
        if skip_llm:
            answer_text = INSUFFICIENT_NOTE
        else:
            self._emit_status("Сборка ответа моделью…")
            try:
                answer_text, model, tin, tout, cost, llm_ms = self._synthesize(
                    objective,
                    hits,
                    partial=self.limits.usage.partial,
                    table_md=table_md if isinstance(table_md, str) else None,
                    web_snippets=snippets,
                )
            except Exception as exc:  # noqa: BLE001 — keep the tool result, do not retry it
                return self._fail_kept(run, workflow, str(exc), stage="synthesize", t0=t0)
            self.limits.note_llm(tin + tout, cost)
            self._record_step(
                "synthesize", {"revision": False, "resumed": resumed}, {"model": model}
            )

        critic_report: dict[str, Any] | None = None
        need_critic = (not workflow.get("critic_done")) and (
            complexity in ("verify", "research", "multi") or "critic" in roles
        )
        if need_critic and not self.limits.would_exceed():
            self._emit_status("Проверка опоры на источники (critic)…")
            critic_report = run_critic(
                gate=self.gate,
                registry=self.registry,
                limits=self.limits,
                objective=objective,
                answer=answer_text,
                sources=source_texts,
                run=run,
            )
            workflow["critic_done"] = True
            self._record_step(
                "tool", {"tool": "verify_citations", "resumed": resumed}, critic_report
            )
            if (
                not skip_llm
                and critic_report.get("needs_revision")
                and not self.limits.would_exceed()
            ):
                self._emit_status("Повторная сборка ответа после критика…")
                try:
                    answer_text, model2, tin2, tout2, cost2, llm_ms2 = self._synthesize(
                        objective + "\n\nИсправь неподтверждённые утверждения.",
                        hits,
                        partial=self.limits.usage.partial,
                        table_md=table_md if isinstance(table_md, str) else None,
                        web_snippets=snippets,
                        revision_note="critic requested revision",
                    )
                except Exception as exc:  # noqa: BLE001
                    return self._fail_kept(run, workflow, str(exc), stage="synthesize", t0=t0)
                model = model2 or model
                tin += tin2
                tout += tout2
                cost += cost2
                llm_ms = llm_ms2 or llm_ms
                self.limits.note_llm(tin2 + tout2, cost2)
                self._record_step(
                    "synthesize", {"revision": True, "resumed": resumed}, {"model": model}
                )

        grounding = check_grounding(answer_text, source_texts)
        if critic_report and (
            critic_report.get("grounding_ok") is False or critic_report.get("needs_revision")
        ):
            grounding_ok = False
        else:
            grounding_ok = grounding.ok

        if isinstance(table_md, str) and "Aspect" in table_md and table_md not in answer_text:
            answer_text = f"{answer_text.rstrip()}\n\n{table_md}"
        if note and not answer_text.startswith(note):
            answer_text = f"{note}\n{answer_text}"
        if self.limits.usage.partial and not answer_text.startswith("[частичный"):
            answer_text = "[частичный ответ — бюджет исчерпан]\n" + answer_text

        distill_info = None
        if self.settings.distill_enabled and self.registry.get("distill_run"):
            dtool = self.registry.get("distill_run")
            assert dtool is not None
            dres = self.gate.execute(
                run=run,
                tool=dtool,
                args={
                    "run_id": str(run.id),
                    "objective": objective,
                    "answer": answer_text[:2000],
                    "citations": citations,
                    "plan": plan,
                },
            )
            if dres.executed:
                distill_info = dres.output
                self._record_step("tool", {"tool": "distill_run"}, distill_info)

        self._note_slice(t0)
        get_bus().post(
            f"run:{run.id}",
            "completed",
            sender="orchestrator",
            meta={"grounding_ok": grounding_ok},
        )
        run.state = RunState.COMPLETED
        run.usage = self.limits.usage.to_dict()
        run.result = {
            "answer": answer_text,
            "partial": self.limits.usage.partial,
            "grounding_ok": grounding_ok,
            "citations": citations,
            "hits": hits,
            "critic": critic_report,
            "analyst": analyst_report,
            "web": web_report,
            "distill": distill_info,
            "resumed": resumed,
        }
        self._record_step(
            "synthesize",
            {"partial": self.limits.usage.partial, "resumed": resumed},
            {"model": model, "chars": len(answer_text)},
        )
        self._commit()
        latency = int((perf_counter() - t0) * 1000)
        return AgentAnswer(
            run_id=str(run.id),
            answer=answer_text,
            model=model,
            citations=citations,
            retrieval_mode=str(workflow.get("retrieval_mode") or "agent"),
            grounding_ok=grounding_ok,
            usage=self.limits.usage.to_dict(),
            state=RunState.COMPLETED,
            partial=self.limits.usage.partial,
            plan=plan,
            audit=self._audit_dicts(),
            steps=list(self._steps),
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            latency_ms=latency or llm_ms,
        )

    def _harvest_batch(
        self, agg: dict[str, Any]
    ) -> tuple[
        list[dict[str, Any]],
        str,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        hits: list[dict[str, Any]] = []
        mode = "subagents"
        critic: dict[str, Any] | None = None
        analyst: dict[str, Any] | None = None
        web: dict[str, Any] | None = None
        for child in agg.get("children") or []:
            if child.get("state") == "cancelled":
                continue
            res = child.get("result") or {}
            role = child.get("role")
            if role == "retriever":
                hits = list(res.get("hits") or hits)
                mode = str(res.get("retrieval_mode") or mode)
            elif role == "critic":
                critic = res
                if not hits and res.get("hits"):
                    hits = list(res["hits"])
            elif role == "analyst":
                analyst = res
            elif role == "web":
                web = res
        return hits, mode, critic, analyst, web

    def _tool_spawn(self, *, roles: list[str], objective: str) -> dict[str, Any]:
        parent = self._active_run
        # Shared scratch so analyst can see KB/web from earlier children in same batch
        scratch: dict[str, Any] = {"hits": [], "web_results": []}
        role_tool = {
            "retriever": "search_kb",
            "web": "web_search",
            "analyst": "build_comparison_table",
            "critic": "verify_citations",
        }

        def runner(child: ChildRun) -> dict[str, Any]:
            tool = role_tool.get(child.role, child.role)
            self._emit_status(f"Субагент «{child.role}» ({tool})…")
            if child.role == "retriever":
                out = run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                )
                scratch["hits"] = list(out.get("hits") or [])
                self._record_step("tool", {"tool": tool, "role": child.role}, out)
                return out
            if child.role == "web":
                out = run_web(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                )
                scratch["web_results"] = evidence_snippets(
                    list(out.get("results") or []),
                    list(out.get("pages") or []),
                    mock_mode=bool(self.settings.web_search_mock),
                )
                self._record_step("tool", {"tool": tool, "role": child.role}, out)
                return out
            if child.role == "analyst":
                out = run_analyst(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                    kb_hits=list(scratch.get("hits") or []),
                    web_results=list(scratch.get("web_results") or []),
                )
                self._record_step("tool", {"tool": tool, "role": child.role}, out)
                return out
            if child.role == "critic":
                hits = list(scratch.get("hits") or [])
                web = list(scratch.get("web_results") or [])
                if not hits and not web:
                    ret = run_retriever(
                        gate=self.gate,
                        registry=self.registry,
                        limits=self.limits,
                        objective=objective,
                        run=parent,
                    )
                    hits = list(ret.get("hits") or [])
                    scratch["hits"] = hits
                _, source_texts, _ = self._merge_sources(hits, web)
                draft = self._draft_from_hits(objective, hits, web)
                report = run_critic(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    answer=draft,
                    sources=source_texts,
                    run=parent,
                )
                report["hits"] = hits
                report["web_results"] = web
                self._record_step("tool", {"tool": tool, "role": child.role}, report)
                return report
            return {"error": f"unknown_role:{child.role}"}

        return self.subagents.spawn(
            parent_id=getattr(parent, "id", None),
            roles=roles,
            objective=objective,
            runner=runner,
        )

    def _tool_status(self, *, batch_id: str | None = None) -> dict[str, Any]:
        return self.subagents.status(batch_id)

    def _tool_followup(self, *, child_id: int, message: str) -> dict[str, Any]:
        return self.subagents.followup(child_id, message)

    def _tool_stop(self, *, child_id: int) -> dict[str, Any]:
        return self.subagents.stop(child_id)

    @staticmethod
    def _merge_sources(
        hits: list[dict[str, Any]],
        web_snippets: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        """KB first, then web — same order as prompt `[n]` / critic / UI citations."""
        contexts: list[dict[str, Any]] = []
        texts: list[str] = []
        citations: list[dict[str, Any]] = []
        n = 0
        for h in hits:
            n += 1
            text = h.get("text") or ""
            contexts.append(
                {
                    "text": text,
                    "document_name": h.get("document_name"),
                    "score": h.get("score"),
                    "section_path": h.get("section_path"),
                }
            )
            texts.append(text)
            citations.append(
                {
                    "n": n,
                    "kind": "kb",
                    "chunk_id": h.get("chunk_id"),
                    "doc_id": h.get("document_id"),
                    "title": h.get("document_name") or "",
                    "section_path": h.get("section_path"),
                    "page": h.get("page"),
                    "score": float(h.get("score") or 0.0),
                    "text": text,
                    "url": None,
                }
            )
        for w in web_snippets or []:
            n += 1
            text = w.get("snippet") or w.get("text") or ""
            url = (w.get("url") or "").strip() or None
            title = w.get("title") or f"web-{n}"
            contexts.append(
                {
                    "text": text,
                    "document_name": title,
                    "score": 0.0,
                    "section_path": url,
                }
            )
            texts.append(text)
            citations.append(
                {
                    "n": n,
                    "kind": "web",
                    "chunk_id": None,
                    "doc_id": None,
                    "title": title,
                    "section_path": url,
                    "page": None,
                    "score": 0.0,
                    "text": text,
                    "url": url,
                    "origin": w.get("origin"),
                }
            )
        return contexts, texts, citations

    def _draft_from_hits(
        self,
        objective: str,
        hits: list[dict[str, Any]],
        web: list[dict[str, Any]] | None = None,
    ) -> str:
        if not hits and not web:
            return f"Нет контекста для: {objective}"
        _, _, cites = self._merge_sources(hits, web)
        if not cites:
            return f"Нет контекста для: {objective}"
        top = cites[0]
        snippet = (top.get("text") or "")[:200]
        return f"По источникам: {snippet} [1]"

    def _synthesize(
        self,
        objective: str,
        hits: list[dict[str, Any]],
        *,
        partial: bool,
        table_md: str | None = None,
        web_snippets: list[dict[str, Any]] | None = None,
        revision_note: str | None = None,
    ) -> tuple[str, str, int, int, float, int]:
        contexts, _, _ = self._merge_sources(hits, web_snippets)
        prompt = build_rag_prompt(objective, contexts)
        if table_md:
            prompt += f"\n\nЧерновик таблицы от Analyst:\n{table_md}\nВключи таблицу в ответ."
        if revision_note:
            prompt += f"\n\n({revision_note})"
        if partial:
            prompt += "\n\n(Бюджет агента исчерпан — дай краткий частичный ответ.)"

        if callable(self._llm) and not isinstance(self._llm, LLMGateway):
            result = self._llm(user_message=prompt, system_prompt=SYSTEM_PROMPT)
        else:
            gateway = self._llm if isinstance(self._llm, LLMGateway) else LLMGateway(self.settings)
            try:
                result = gateway.chat(user_message=prompt, system_prompt=SYSTEM_PROMPT)
            except Exception:
                text = self._heuristic_answer(objective, hits, partial=partial, table_md=table_md)
                return text, "heuristic", 0, 0, 0.0, 0

        return (
            result.text,
            result.model,
            result.tokens_in,
            result.tokens_out,
            result.cost_usd,
            result.latency_ms,
        )

    @staticmethod
    def _heuristic_answer(
        objective: str,
        hits: list[dict[str, Any]],
        *,
        partial: bool,
        table_md: str | None = None,
    ) -> str:
        if not hits and not table_md:
            msg = "В базе нет релевантного контекста для ответа."
            return f"[частичный ответ — бюджет исчерпан]\n{msg}" if partial else msg
        if hits:
            top = hits[0]
            snippet = (top.get("text") or "")[:400]
            title = top.get("document_name") or "источник"
            body = f"Согласно [{title}][1]: {snippet}"
        else:
            body = f"Сводка по запросу: {objective}"
        if table_md:
            body = f"{body}\n\n{table_md}"
        if partial:
            return f"[частичный ответ — бюджет исчерпан]\n{body}"
        return body

    def _emit_status(self, text: str) -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(text)
        except Exception:  # noqa: BLE001 — UI callback must not kill the run
            pass

    def _record_step(self, kind: str, inp: dict[str, Any], out: Any) -> None:
        self._step_n += 1
        rec = {"n": self._step_n, "kind": kind, "input": inp, "output": out}
        self._steps.append(rec)
        if self.persist and self.db is not None and self._active_run is not None:
            row_id = getattr(self._active_run, "id", None)
            if isinstance(row_id, int):
                self.db.add(
                    Step(
                        run_id=row_id,
                        n=self._step_n,
                        kind=kind,
                        input=inp,
                        output=out if isinstance(out, dict) else {"value": out},
                    )
                )
                # Flush so UI can see steps while the run is still in progress
                try:
                    self.db.flush()
                except Exception:  # noqa: BLE001
                    pass
        if self._on_step is not None:
            try:
                self._on_step(rec)
            except Exception:  # noqa: BLE001 — UI callback must not kill the run
                pass

    def _commit(self) -> None:
        if self.persist and self.db is not None:
            self.db.commit()

    def _audit_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "ts": e.ts,
                "tool": e.tool,
                "rule_id": e.rule_id,
                "mode": e.mode,
                "decision": e.decision,
                "args": e.args,
                "error": e.error,
            }
            for e in self.audit.events
        ]
