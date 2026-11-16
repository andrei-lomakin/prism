"""AgentRuntime: triage → plan → act (via ActionGate) → synthesize."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from prism.agents.memory import MemoryStore
from prism.agents.policy import ActionGate, PolicyBundle, load_policy
from prism.agents.policy.audit import AuditLog
from prism.agents.roles import run_critic, run_retriever
from prism.agents.runtime.limits import LimitTracker, RunBudget
from prism.agents.runtime.subagents import ChildRun, SubagentManager
from prism.agents.runtime.triage import TriageResult, triage
from prism.agents.tools import ToolContext, ToolRegistry, build_default_registry
from prism.config import Settings, get_settings
from prism.db.models import AgentRun, Step
from prism.llm import LLMGateway
from prism.llm.gateway import LLMResult
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
        persist: bool = True,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.budget = budget or RunBudget()
        self.audit = AuditLog()
        self.policy = policy or load_policy()
        self.gate = ActionGate(self.policy, audit=self.audit, approval_fn=approval_fn)
        self.memory = MemoryStore(db)
        self.subagents = SubagentManager()
        self.limits = LimitTracker(self.budget)
        self.persist = persist and db is not None
        self._llm = llm
        self._steps: list[dict[str, Any]] = []
        self._step_n = 0

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
                budget={
                    "max_steps": self.budget.max_steps,
                    "max_searches": self.budget.max_searches,
                    "max_tokens": self.budget.max_tokens,
                    "max_cost": self.budget.max_cost_usd,
                },
                usage={},
            )
            self.db.add(row)
            self.db.flush()
            return row
        return _EphemeralRun(role=role, objective=objective)

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
        awaiting = False

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
            elif role == "critic":
                # Critic alone: search then verify empty draft signal
                ret = run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=run,
                )
                hits = list(ret.get("hits") or [])
                draft = self._draft_from_hits(objective, hits)
                critic_report = run_critic(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    answer=draft,
                    sources=[h.get("text", "") for h in hits],
                    run=run,
                )
                self._record_step("tool", {"tool": "verify_citations"}, critic_report)
            else:
                # Orchestrator
                if tri.complexity == "multi":
                    run.state = RunState.AWAITING_SUBAGENTS
                    self._commit()
                    spawn_tool = self.registry.get("spawn_subagents")
                    assert spawn_tool is not None
                    self.limits.note_step()
                    gr = self.gate.execute(
                        run=run,
                        tool=spawn_tool,
                        args={"roles": tri.roles, "objective": objective},
                    )
                    self._record_step("delegation", {"roles": tri.roles}, gr.output)
                    if not gr.executed:
                        if gr.decision.mode in ("ask", "on_request"):
                            run.state = RunState.AWAITING_APPROVAL
                            awaiting = True
                        else:
                            self.limits.mark_forced()
                    else:
                        agg = gr.output or {}
                        hits, retrieval_mode, critic_report = self._harvest_batch(agg)
                        run.state = RunState.RUNNING
                else:
                    # simple / verify: direct search via gate
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

            if awaiting:
                latency = int((perf_counter() - t0) * 1000)
                return AgentAnswer(
                    run_id=str(run.id),
                    answer="Требуется одобрение действия (ActionGate ask).",
                    model="none",
                    citations=[],
                    retrieval_mode=retrieval_mode,
                    grounding_ok=False,
                    usage=self.limits.usage.to_dict(),
                    state=RunState.AWAITING_APPROVAL,
                    plan=plan,
                    audit=self._audit_dicts(),
                    steps=list(self._steps),
                    latency_ms=latency,
                )

            if self.limits.would_exceed():
                self.limits.mark_forced()

            answer_text, model, tin, tout, cost, llm_ms = self._synthesize(
                objective, hits, partial=self.limits.usage.partial
            )
            self.limits.note_llm(tin + tout, cost)

            if tri.complexity == "verify" or critic_report is None and hits:
                # Optional critic pass when budget allows
                if not self.limits.would_exceed() and (
                    tri.complexity == "verify" or "critic" in tri.roles
                ):
                    critic_report = run_critic(
                        gate=self.gate,
                        registry=self.registry,
                        limits=self.limits,
                        objective=objective,
                        answer=answer_text,
                        sources=[h.get("text", "") for h in hits],
                        run=run,
                    )
                    self._record_step("tool", {"tool": "verify_citations"}, critic_report)

            grounding = check_grounding(answer_text, [h.get("text", "") for h in hits])
            if critic_report and critic_report.get("grounding_ok") is False:
                grounding_ok = False
            else:
                grounding_ok = grounding.ok

            if self.limits.usage.partial and not answer_text.startswith("[частичный"):
                answer_text = "[частичный ответ — бюджет исчерпан]\n" + answer_text

            citations = [
                {
                    "n": i,
                    "chunk_id": h.get("chunk_id"),
                    "doc_id": h.get("document_id"),
                    "title": h.get("document_name") or "",
                    "section_path": h.get("section_path"),
                    "page": h.get("page"),
                    "score": float(h.get("score") or 0.0),
                }
                for i, h in enumerate(hits, start=1)
            ]

            run.state = RunState.COMPLETED
            run.usage = self.limits.usage.to_dict()
            run.result = {
                "answer": answer_text,
                "partial": self.limits.usage.partial,
                "grounding_ok": grounding_ok,
                "critic": critic_report,
            }
            self._record_step(
                "synthesize",
                {"partial": self.limits.usage.partial},
                {"model": model, "chars": len(answer_text)},
            )
            self._commit()

            latency = int((perf_counter() - t0) * 1000)
            return AgentAnswer(
                run_id=str(run.id),
                answer=answer_text,
                model=model,
                citations=citations,
                retrieval_mode=retrieval_mode,
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
        except Exception as exc:  # noqa: BLE001
            run.state = RunState.FAILED
            run.result = {"error": str(exc)}
            self._commit()
            raise

    def _harvest_batch(
        self, agg: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
        hits: list[dict[str, Any]] = []
        mode = "subagents"
        critic: dict[str, Any] | None = None
        for child in agg.get("children") or []:
            res = child.get("result") or {}
            if child.get("role") == "retriever":
                hits = list(res.get("hits") or hits)
                mode = str(res.get("retrieval_mode") or mode)
            elif child.get("role") == "critic":
                critic = res
                if not hits and res.get("hits"):
                    hits = list(res["hits"])
        return hits, mode, critic

    def _tool_spawn(self, *, roles: list[str], objective: str) -> dict[str, Any]:
        parent = self._active_run

        def runner(child: ChildRun) -> dict[str, Any]:
            # Nested lightweight run sharing gate/registry/limits
            if child.role == "retriever":
                return run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                )
            if child.role == "critic":
                # Critic child: retrieve + draft + verify
                ret = run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                )
                hits = list(ret.get("hits") or [])
                draft = self._draft_from_hits(objective, hits)
                report = run_critic(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    answer=draft,
                    sources=[h.get("text", "") for h in hits],
                    run=parent,
                )
                report["hits"] = hits
                report["retrieval_mode"] = ret.get("retrieval_mode")
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

    def _draft_from_hits(self, objective: str, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return f"Нет контекста для: {objective}"
        bits = " ".join((h.get("text") or "")[:200] for h in hits[:3])
        return f"По источникам: {bits} [1]"

    def _synthesize(
        self,
        objective: str,
        hits: list[dict[str, Any]],
        *,
        partial: bool,
    ) -> tuple[str, str, int, int, float, int]:
        contexts = [
            {
                "text": h.get("text"),
                "document_name": h.get("document_name"),
                "score": h.get("score"),
                "section_path": h.get("section_path"),
            }
            for h in hits
        ]
        prompt = build_rag_prompt(objective, contexts)
        if partial:
            prompt += "\n\n(Бюджет агента исчерпан — дай краткий частичный ответ.)"

        if callable(self._llm) and not isinstance(self._llm, LLMGateway):
            result = self._llm(user_message=prompt, system_prompt=SYSTEM_PROMPT)
        else:
            gateway = self._llm if isinstance(self._llm, LLMGateway) else LLMGateway(self.settings)
            try:
                result = gateway.chat(user_message=prompt, system_prompt=SYSTEM_PROMPT)
            except Exception:
                # Offline / mock fallback for tests & diploma demos without Ollama
                text = self._heuristic_answer(objective, hits, partial=partial)
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
    def _heuristic_answer(objective: str, hits: list[dict[str, Any]], *, partial: bool) -> str:
        if not hits:
            msg = "В базе нет релевантного контекста для ответа."
            return f"[частичный ответ — бюджет исчерпан]\n{msg}" if partial else msg
        top = hits[0]
        snippet = (top.get("text") or "")[:400]
        title = top.get("document_name") or "источник"
        body = f"Согласно [{title}][1]: {snippet}"
        if partial:
            return f"[частичный ответ — бюджет исчерпан]\n{body}"
        return body

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
