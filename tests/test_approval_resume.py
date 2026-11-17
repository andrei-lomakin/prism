"""Approve continues the same run and does not repeat a finished call."""

import time

from prism.agents.policy.approvals import ApprovalStore
from prism.agents.runtime import AgentRuntime
from prism.config import get_settings
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


def test_resume_runs_approved_action_on_same_run():
    store = ApprovalStore(ttl_seconds=60)
    settings = get_settings().model_copy(
        update={"space_filter": "project", "distill_enabled": False}
    )
    rt = AgentRuntime(db=None, llm=_mock_llm, persist=False, approvals=store, settings=settings)
    seen: list[str] = []
    tool = rt.registry.get("web_search")
    assert tool is not None

    def wrapped(query: str) -> dict:
        seen.append(query)
        return {
            "query": query,
            "results": [
                {
                    "title": "T",
                    "url": "https://example.com/a",
                    "snippet": "approved snippet",
                    "origin": "mock",
                    "provider": "mock",
                }
            ],
            "origin": "mock",
            "provider": "mock",
            "ok": True,
            "status": "ok",
        }

    tool.execute = wrapped
    objective = "найди в вебе bm25"
    paused = rt.handle(objective, role="web")
    assert paused.state == "awaiting_approval"
    assert seen == []
    assert store.pending()[0].args == {"query": objective}
    before = paused.usage["active_seconds"]

    try:
        rt.resume(paused.run_id)
        raise AssertionError("resume before approve must fail")
    except RuntimeError as exc:
        assert "not active" in str(exc)
    assert seen == []

    time.sleep(0.6)
    store.approve(store.pending()[0].id)
    mid = rt.resume(paused.run_id)
    assert mid.run_id == paused.run_id
    assert mid.state == "awaiting_approval"
    assert seen == [objective]
    assert any(
        event["tool"] == "web_search" and event["decision"] == "allow" for event in mid.audit
    )
    pending = store.pending()[0]
    assert pending.tool == "fetch_page"
    assert pending.args == {"url": "https://example.com/a"}

    try:
        rt.resume(paused.run_id)
        raise AssertionError("second resume must not run fetch before it is approved")
    except RuntimeError as exc:
        assert "not active" in str(exc)
    assert seen == [objective]

    store.approve(pending.id)
    rt3 = AgentRuntime(db=None, llm=_mock_llm, persist=False, approvals=store)
    done = rt3.resume(paused.run_id)
    assert rt3.settings.space_filter == "project"
    assert done.run_id == paused.run_id
    assert done.state == "completed"
    assert seen == [objective]
    assert any("Mock page body" in (item.get("text") or "") for item in done.citations)
    assert "тестовые" in done.answer
    assert done.usage["steps"] > paused.usage["steps"]
    assert done.usage["active_seconds"] >= before
    assert done.usage["active_seconds"] - before < 0.45

    try:
        rt3.resume(paused.run_id)
        raise AssertionError("completed run must not execute again")
    except RuntimeError:
        pass
    assert seen == [objective]
