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
from prism.agents.roles import run_analyst, run_critic, run_retriever, run_web
from prism.agents.runtime.bus import get_bus
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
        on_step: Callable[[dict[str, Any]], None] | None = None,
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
        self._on_step = on_step
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

            if awaiting:
                latency = int((perf_counter() - t0) * 1000)
                msg = "Требуется одобрение действия (ActionGate ask)."
                if approval_id:
                    msg += f" approval_id={approval_id}"
                return AgentAnswer(
                    run_id=str(run.id),
                    answer=msg,
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

            # Prefer analyst table as scaffolding for synthesis
            table_md = (analyst_report or {}).get("markdown")
            web_snippets = list((web_report or {}).get("results") or [])
            _, source_texts, citations = self._merge_sources(hits, web_snippets)
            answer_text, model, tin, tout, cost, llm_ms = self._synthesize(
                objective,
                hits,
                partial=self.limits.usage.partial,
                table_md=table_md if isinstance(table_md, str) else None,
                web_snippets=web_snippets,
            )
            self.limits.note_llm(tin + tout, cost)

            if not self.limits.would_exceed() and (
                tri.complexity in ("verify", "research", "multi") or "critic" in tri.roles
            ):
                critic_report = run_critic(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    answer=answer_text,
                    sources=source_texts,
                    run=run,
                )
                self._record_step("tool", {"tool": "verify_citations"}, critic_report)
                if critic_report.get("needs_revision") and not self.limits.would_exceed():
                    answer_text, model2, tin2, tout2, cost2, llm_ms2 = self._synthesize(
                        objective + "\n\nИсправь неподтверждённые утверждения.",
                        hits,
                        partial=self.limits.usage.partial,
                        table_md=table_md if isinstance(table_md, str) else None,
                        web_snippets=web_snippets,
                        revision_note="critic requested revision",
                    )
                    model = model2 or model
                    tin += tin2
                    tout += tout2
                    cost += cost2
                    llm_ms = llm_ms2 or llm_ms
                    self.limits.note_llm(tin2 + tout2, cost2)
                    self._record_step("synthesize", {"revision": True}, {"model": model})

            grounding = check_grounding(answer_text, source_texts)
            if critic_report and (
                critic_report.get("grounding_ok") is False or critic_report.get("needs_revision")
            ):
                grounding_ok = False
            else:
                grounding_ok = grounding.ok

            if table_md and "Aspect" in table_md and table_md not in answer_text:
                answer_text = f"{answer_text.rstrip()}\n\n{table_md}"

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
                "critic": critic_report,
                "analyst": analyst_report,
                "web": web_report,
                "distill": distill_info,
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

        def runner(child: ChildRun) -> dict[str, Any]:
            if child.role == "retriever":
                out = run_retriever(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                )
                scratch["hits"] = list(out.get("hits") or [])
                return out
            if child.role == "web":
                out = run_web(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                )
                scratch["web_results"] = list(out.get("results") or [])
                return out
            if child.role == "analyst":
                return run_analyst(
                    gate=self.gate,
                    registry=self.registry,
                    limits=self.limits,
                    objective=objective,
                    run=parent,
                    kb_hits=list(scratch.get("hits") or []),
                    web_results=list(scratch.get("web_results") or []),
                )
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
