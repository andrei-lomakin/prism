"""AgentRuntime: dedup, subagents aggregate, forced partial, mock-LLM path."""

from __future__ import annotations

from prism.agents.runtime import AgentRuntime, RunBudget, SubagentManager, triage
from prism.agents.runtime.limits import LimitTracker
from prism.agents.runtime.subagents import ChildRun
from prism.llm.gateway import LLMResult


def _mock_llm(*, user_message: str, system_prompt: str) -> LLMResult:
    return LLMResult(
        text="Ответ по контексту [1].",
        model="mock",
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.0,
        latency_ms=1,
    )


def test_triage_simple_vs_multi():
    s = triage("Что такое RRF?")
    assert s.complexity == "simple"
    m = triage("Сравни dense и hybrid retrieval")
    assert m.complexity == "multi"
    assert "critic" in m.roles


def test_dedup_read_calls():
    lim = LimitTracker(RunBudget())
    args = {"query": "same"}
    assert lim.is_duplicate_read("search_kb", args, side_effect="read") is False
    assert lim.is_duplicate_read("search_kb", args, side_effect="read") is True
    assert lim.usage.skipped_dedup == 1
    # write not deduped by fingerprint store the same way for side_effect write
    assert lim.is_duplicate_read("remember", {"k": 1}, side_effect="write") is False
    assert lim.is_duplicate_read("remember", {"k": 1}, side_effect="write") is False


def test_budget_forced_partial_answer():
    rt = AgentRuntime(
        db=None,
        llm=_mock_llm,
        budget=RunBudget(max_steps=0, max_searches=0),
        persist=False,
    )
    # max_steps=0 → would_exceed immediately → forced synthesis
    ans = rt.handle("Что такое pgvector?")
    assert ans.state == "completed"
    assert ans.partial is True
    assert "частичный" in ans.answer.lower()


def test_simple_agent_run_mock_llm():
    rt = AgentRuntime(db=None, llm=_mock_llm, persist=False)
    ans = rt.handle("Что такое BM25?")
    assert ans.state == "completed"
    assert ans.model == "mock"
    assert ans.plan["complexity"] == "simple"
    assert any(s["kind"] == "tool" for s in ans.steps)
    assert any(e["tool"] == "search_kb" and e["decision"] == "allow" for e in ans.audit)


def test_parent_awaits_children_aggregate():
    mgr = SubagentManager()
    seen: list[str] = []

    def runner(child: ChildRun) -> dict:
        seen.append(child.role)
        if child.role == "retriever":
            return {"hits": [{"chunk_id": 1, "text": "A", "score": 1.0}], "retrieval_mode": "mock"}
        return {"grounding_ok": True, "role": "critic"}

    agg = mgr.spawn(
        parent_id="p1",
        roles=["retriever", "critic"],
        objective="compare X and Y",
        runner=runner,
    )
    assert agg["done"] is True
    assert {c["role"] for c in agg["children"]} == {"retriever", "critic"}
    assert all(c["state"] == "completed" for c in agg["children"])
    assert seen == ["retriever", "critic"]

    status = mgr.status(agg["batch_id"])
    assert status["done"] is True


def test_multi_orchestrator_spawns_subagents():
    rt = AgentRuntime(db=None, llm=_mock_llm, persist=False)
    ans = rt.handle("Сравни dense и lexical поиск по метрикам")
    assert ans.state == "completed"
    assert ans.plan["complexity"] == "multi"
    assert any(s["kind"] == "delegation" for s in ans.steps)
    # spawn + child searches should appear in audit
    tools = [e["tool"] for e in ans.audit if e["decision"] == "allow"]
    assert "spawn_subagents" in tools
    assert "search_kb" in tools


def test_memory_remember_recall():
    from prism.agents.memory import MemoryStore

    mem = MemoryStore(db=None)
    mid = mem.remember("journal", "pref", "prefer hybrid")
    assert mid > 0
    hits = mem.recall("hybrid")
    assert len(hits) == 1
    assert hits[0].content == "prefer hybrid"
    # soft supersede
    mem.remember("journal", "pref", "prefer RRF")
    hits2 = mem.recall("pref")
    assert len(hits2) == 1
    assert hits2[0].content == "prefer RRF"
