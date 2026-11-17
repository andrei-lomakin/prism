"""Approve continues the same run and executes the original tool args."""

from prism.agents.policy.approvals import ApprovalStore
from prism.agents.runtime import AgentRuntime
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
    rt = AgentRuntime(db=None, llm=_mock_llm, persist=False, approvals=store)
    seen: list[str] = []
    tool = rt.registry.get("web_search")
    assert tool is not None

    def wrapped(query: str) -> dict:
        seen.append(query)
        return {
            "query": query,
            "results": [
                {"title": "T", "url": "https://example.com/a", "snippet": "approved snippet"}
            ],
        }

    tool.execute = wrapped
    objective = "найди в вебе bm25"
    paused = rt.handle(objective, role="web")
    assert paused.state == "awaiting_approval"
    assert seen == []
    assert store.pending()[0].args == {"query": objective}

    try:
        rt.resume(paused.run_id)
        raise AssertionError("resume before approve must fail")
    except RuntimeError as exc:
        assert "not active" in str(exc)

    store.approve(store.pending()[0].id)
    done = rt.resume(paused.run_id)
    assert done.run_id == paused.run_id
    assert done.state == "completed"
    assert seen == [objective]
    assert any(
        event["tool"] == "web_search" and event["decision"] == "allow" for event in done.audit
    )
