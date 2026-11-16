from prism.agents.memory import MemoryStore
from prism.agents.memory.distill import distill_run_to_memory
from prism.agents.runtime import AgentRuntime, RunBudget
from prism.agents.runtime.bus import MessageBus
from prism.agents.runtime.triage import triage
from prism.llm.gateway import LLMResult


def _mock_llm(*, user_message: str, system_prompt: str) -> LLMResult:
    return LLMResult(text="Summary [1].", model="mock", tokens_in=1, tokens_out=1)


def test_triage_research_includes_web_analyst():
    # UTF-8 explicit: comparative + "statyam" stem / web markers
    q = "Sravni dense i hybrid po baze i svezhim statyam, svedi v tablitsu"
    # Use real RU string via unicode escapes to avoid Windows encoding issues
    q_ru = (
        "\u0421\u0440\u0430\u0432\u043d\u0438 dense \u0438 hybrid "
        "\u043f\u043e \u0431\u0430\u0437\u0435 \u0438 \u0441\u0432\u0435\u0436\u0438\u043c "
        "\u0441\u0442\u0430\u0442\u044c\u044f\u043c, \u0441\u0432\u0435\u0434\u0438 "
        "\u0432 \u0442\u0430\u0431\u043b\u0438\u0446\u0443"
    )
    t = triage(q_ru)
    assert t.complexity == "research"
    assert "web" in t.roles and "analyst" in t.roles and "critic" in t.roles
    assert triage(q).complexity in ("multi", "simple", "research")


def test_distill_soft_supersede():
    mem = MemoryStore(None)
    a = distill_run_to_memory(
        mem, run_id="1", objective="q", answer="- fact one\n- fact two", citations=[]
    )
    b = distill_run_to_memory(mem, run_id="1", objective="q", answer="- fact three", citations=[])
    assert a["key"] == b["key"]
    hits = mem.recall("run:1")
    assert len(hits) == 1
    assert "fact three" in hits[0].content


def test_message_bus_urgent_wakeup():
    bus = MessageBus()
    assert bus.wait("c1", timeout=0) is False
    bus.post("c1", "hi", urgent=True)
    assert bus.wait("c1", timeout=0.1) is True
    assert bus.read("c1")[0].body == "hi"


def test_research_run_with_auto_approve_web():
    rt = AgentRuntime(
        db=None,
        llm=_mock_llm,
        persist=False,
        budget=RunBudget(max_steps=20, max_searches=10),
        approval_fn=lambda *_a, **_k: True,
    )
    q = (
        "\u0421\u0440\u0430\u0432\u043d\u0438 BM25 \u0438 dense "
        "\u043f\u043e \u0431\u0430\u0437\u0435 \u0438 web "
        "\u0441\u0442\u0430\u0442\u044c\u044f\u043c, "
        "\u0442\u0430\u0431\u043b\u0438\u0446\u0430"
    )
    ans = rt.handle(q)
    assert ans.state == "completed"
    assert ans.plan["complexity"] == "research"
    assert any(s["kind"] == "delegation" for s in ans.steps)
    tools = [e["tool"] for e in ans.audit if e.get("decision") == "allow"]
    assert "web_search" in tools
    assert "build_comparison_table" in tools or "spawn_subagents" in tools
