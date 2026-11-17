"""R1–R4: resume continues the workflow, errors stay failed, live web is not mocked."""

from __future__ import annotations

import time

import httpx
import pytest

from prism.agents.policy.approvals import ApprovalStore
from prism.agents.runtime import AgentRuntime, RunBudget
from prism.agents.runtime.limits import LimitTracker
from prism.agents.runtime.runtime import _EPHEMERAL_RUNS
from prism.agents.web_evidence import evidence_snippets
from prism.api.routes.runs import _resume_approved
from prism.config import get_settings
from prism.db.models import AgentRun, Step
from prism.llm.gateway import LLMResult

RESEARCH = (
    "\u0421\u0440\u0430\u0432\u043d\u0438 BM25 \u0438 dense "
    "\u043f\u043e \u0431\u0430\u0437\u0435 \u0438 web "
    "\u0441\u0442\u0430\u0442\u044c\u044f\u043c, "
    "\u0442\u0430\u0431\u043b\u0438\u0446\u0430"
)


def _llm_text(
    text: str = "\u041e\u0442\u0432\u0435\u0442 \u043f\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0443 [1].",
):
    def _call(*, user_message: str, system_prompt: str) -> LLMResult:
        return LLMResult(
            text=text, model="mock", tokens_in=4, tokens_out=3, cost_usd=0.0, latency_ms=1
        )

    return _call


class _Recorder:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, *, user_message: str, system_prompt: str) -> LLMResult:
        self.prompts.append(user_message)
        return LLMResult(
            text="\u041e\u0442\u0432\u0435\u0442 \u043f\u043e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0443 [1].",
            model="mock",
            tokens_in=4,
            tokens_out=3,
            cost_usd=0.0,
            latency_ms=1,
        )


class _Query:
    def __init__(self, db: FakeDB) -> None:
        self.db = db

    def filter(self, *args, **kwargs):  # noqa: ANN002
        return self

    def filter_by(self, **kwargs):  # noqa: ANN003
        return self

    def order_by(self, *args):  # noqa: ANN002
        return self

    def first(self):  # noqa: ANN201
        return None

    def all(self):  # noqa: ANN201
        return []

    def delete(self, *args, **kwargs):  # noqa: ANN002
        return 0

    def scalar(self):  # noqa: ANN201
        nums = [step.n for step in self.db.steps]
        return max(nums) if nums else None


class FakeDB:
    def __init__(self) -> None:
        self.runs: dict[int, AgentRun] = {}
        self.steps: list[Step] = []
        self.seq = 0

    def add(self, obj: object) -> None:
        name = type(obj).__name__
        if name == "AgentRun":
            if getattr(obj, "id", None) is None:
                self.seq += 1
                obj.id = self.seq  # type: ignore[attr-defined]
            self.runs[int(obj.id)] = obj  # type: ignore[attr-defined]
        elif name == "Step":
            self.steps.append(obj)  # type: ignore[arg-type]
        elif getattr(obj, "id", None) is None:
            self.seq += 1
            obj.id = self.seq  # type: ignore[attr-defined]

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def get(self, model: type, ident: int):  # noqa: ANN201
        if getattr(model, "__name__", "") == "AgentRun":
            return self.runs.get(int(ident))
        return None

    def query(self, *args, **kwargs):  # noqa: ANN002
        return _Query(self)


def _kb_hit(query: str, top_k: int | None = None) -> dict:
    return {
        "hits": [
            {
                "chunk_id": 7,
                "document_id": 3,
                "document_name": "kb.md",
                "text": "KB-UNIQUE-ALPHA",
                "score": 0.91,
                "section_path": "s",
                "page": 1,
            }
        ],
        "retrieval_mode": "mock",
        "query": query,
    }


def _allows(runtimes: list[AgentRuntime], tool: str) -> int:
    return sum(
        1
        for runtime in runtimes
        for event in runtime.audit.events
        if event.tool == tool and event.decision == "allow"
    )


def test_api_resume_new_runtime_continues_research(monkeypatch):
    store = ApprovalStore(ttl_seconds=120)
    recorder = _Recorder()
    db = FakeDB()
    created: list[AgentRuntime] = []
    settings = get_settings().model_copy(
        update={"space_filter": "project", "distill_enabled": False, "web_search_mock": True}
    )
    budget = RunBudget(max_steps=40, max_searches=12)

    class _APIRuntime(AgentRuntime):
        def __init__(self, db=None, *, settings=None, persist=True, **kwargs):  # noqa: ANN001
            super().__init__(
                db,
                settings=settings,
                llm=recorder,
                approvals=store,
                persist=persist,
                # Tiny on purpose: resume must restore the original budget or critic never runs.
                budget=RunBudget(max_steps=2, max_searches=1, max_active_seconds=1),
            )
            created.append(self)

    monkeypatch.setattr("prism.agents.AgentRuntime", _APIRuntime)
    rt = AgentRuntime(
        db,  # type: ignore[arg-type]
        llm=recorder,
        approvals=store,
        persist=True,
        settings=settings,
        budget=budget,
    )
    rt.registry.get("search_kb").execute = _kb_hit  # type: ignore[union-attr]
    paused = rt.handle(RESEARCH)
    assert paused.state == "awaiting_approval"
    row = db.runs[int(paused.run_id)]
    assert row.result["checkpoint"]["hits"][0]["text"] == "KB-UNIQUE-ALPHA"
    paused_steps = row.usage["steps"]
    assert _allows(created, "web_search") == 0

    rec = store.pending()[0]
    assert rec.tool == "web_search"
    store.approve(rec.id)
    first = _resume_approved(db, rec)  # type: ignore[arg-type]
    assert first is not None
    assert first["run_id"] == str(row.id)
    assert first["state"] == "awaiting_approval"
    assert _allows(created, "web_search") == 1
    assert row.result["checkpoint"]["hits"][0]["text"] == "KB-UNIQUE-ALPHA"
    assert created[0].settings.space_filter == "project"
    assert row.result["checkpoint"]["pending"]["tool"] == "fetch_page"

    with pytest.raises(RuntimeError, match="not active"):
        _resume_approved(db, rec)  # type: ignore[arg-type]
    assert _allows(created, "web_search") == 1

    fetches = 0
    for _ in range(4):
        if row.state != "awaiting_approval":
            break
        nxt = store.pending()[0]
        assert nxt.tool == "fetch_page"
        store.approve(nxt.id)
        out = _resume_approved(db, nxt)  # type: ignore[arg-type]
        assert out is not None and out["run_id"] == str(row.id)
        fetches += 1
    assert fetches >= 1
    assert row.state == "completed"
    assert _allows(created, "web_search") == 1
    assert any("KB-UNIQUE-ALPHA" in prompt for prompt in recorder.prompts)
    assert any("Mock page body" in prompt for prompt in recorder.prompts)
    assert "\u0442\u0435\u0441\u0442\u043e\u0432\u044b\u0435" in row.result["answer"]
    cited = " ".join(item["text"] for item in row.result["citations"])
    assert "KB-UNIQUE-ALPHA" in cited
    assert "Mock page body" in cited
    assert "mock_error" not in cited and "mock_fallback" not in cited
    assert row.usage["steps"] > paused_steps
    numbers = [step.n for step in db.steps]
    assert numbers == sorted(numbers)
    assert len(numbers) == len(set(numbers))
    critic = [
        event
        for runtime in created
        for event in runtime.audit.events
        if event.tool == "verify_citations" and event.decision == "allow"
    ]
    assert critic
    blob = str(critic[-1].args)
    assert "KB-UNIQUE-ALPHA" in blob
    assert "Mock page body" in blob


def test_active_seconds_force_partial_budget():
    tracker = LimitTracker(RunBudget(max_active_seconds=1))
    assert tracker.would_exceed() is False
    tracker.note_active(1)
    assert tracker.would_exceed() is True


def test_expired_or_missing_approval_does_not_execute():
    store = ApprovalStore(ttl_seconds=60)
    rt = AgentRuntime(db=None, llm=_llm_text(), persist=False, approvals=store)
    calls: list[str] = []

    def wrapped(query: str) -> dict:
        calls.append(query)
        return {"results": [], "status": "empty", "ok": True, "origin": "live", "provider": "x"}

    rt.registry.get("web_search").execute = wrapped  # type: ignore[union-attr]
    paused = rt.handle(
        "\u043d\u0430\u0439\u0434\u0438 \u0432 \u0432\u0435\u0431\u0435 bm25", role="web"
    )
    assert calls == []
    with pytest.raises(RuntimeError, match="not active"):
        rt.resume(paused.run_id)
    assert calls == []
    rec = store.pending()[0]
    store.approve(rec.id)
    rec.expires_at = time.time() - 1
    with pytest.raises(RuntimeError, match="not active"):
        rt.resume(paused.run_id)
    assert calls == []
    assert _EPHEMERAL_RUNS[paused.run_id].state == "awaiting_approval"


def _boom_runtime(raises: bool):
    store = ApprovalStore(ttl_seconds=60)
    calls: list[int] = []
    rt = AgentRuntime(db=None, llm=_llm_text(), persist=False, approvals=store)

    def wrapped(query: str) -> dict:
        calls.append(1)
        if raises:
            raise RuntimeError("after side effect")
        raise RuntimeError("provider down")

    rt.registry.get("web_search").execute = wrapped  # type: ignore[union-attr]
    paused = rt.handle(
        "\u043d\u0430\u0439\u0434\u0438 \u0432 \u0432\u0435\u0431\u0435 bm25", role="web"
    )
    store.approve(store.pending()[0].id)
    failed = rt.resume(paused.run_id)
    return store, calls, paused, failed


def test_approved_tool_exception_is_not_a_fresh_wait():
    _store, calls, paused, failed = _boom_runtime(False)
    assert failed.state == "failed"
    assert "provider down" in failed.answer
    assert "\u041e\u0434\u043e\u0431\u0440\u0435\u043d\u043e" in failed.answer
    run = _EPHEMERAL_RUNS[paused.run_id]
    assert run.state == "failed"
    assert run.result["checkpoint"] is not None
    assert run.result["failure_stage"] == "tool"
    assert run.result["checkpoint"].get("tool") or run.result["checkpoint"].get("pending")
    with pytest.raises(RuntimeError):
        AgentRuntime(db=None, llm=_llm_text(), persist=False, approvals=_store).resume(
            paused.run_id
        )
    assert calls == [1]


def test_side_effect_then_exception_runs_once():
    store, calls, paused, failed = _boom_runtime(True)
    assert failed.state == "failed"
    assert calls == [1]
    with pytest.raises(RuntimeError):
        AgentRuntime(db=None, llm=_llm_text(), persist=False, approvals=store).resume(paused.run_id)
    assert calls == [1]


def test_synthesis_error_keeps_tool_output_for_retry():
    store = ApprovalStore(ttl_seconds=60)
    calls: list[str] = []

    class _Flaky:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, *, user_message: str, system_prompt: str) -> LLMResult:
            self.n += 1
            if self.n == 1:
                raise RuntimeError("synth broke")
            return LLMResult(
                text="ok [1]", model="mock", tokens_in=1, tokens_out=1, cost_usd=0, latency_ms=1
            )

    flaky = _Flaky()
    rt = AgentRuntime(db=None, llm=flaky, persist=False, approvals=store)

    def wrapped(query: str) -> dict:
        calls.append(query)
        return {
            "query": query,
            "results": [
                {
                    "title": "T",
                    "snippet": "SAVED-SNIPPET",
                    "origin": "mock",
                    "provider": "mock",
                }
            ],
            "origin": "mock",
            "status": "ok",
            "ok": True,
            "provider": "mock",
        }

    rt.registry.get("web_search").execute = wrapped  # type: ignore[union-attr]
    paused = rt.handle(
        "\u043d\u0430\u0439\u0434\u0438 \u0432 \u0432\u0435\u0431\u0435 bm25", role="web"
    )
    store.approve(store.pending()[0].id)
    failed = rt.resume(paused.run_id)
    assert failed.state == "failed"
    assert "synth broke" in failed.answer
    saved = _EPHEMERAL_RUNS[paused.run_id].result["checkpoint"]["saved_output"]
    assert saved["web_results"][0]["snippet"] == "SAVED-SNIPPET"
    assert calls == ["\u043d\u0430\u0439\u0434\u0438 \u0432 \u0432\u0435\u0431\u0435 bm25"]
    again = AgentRuntime(db=None, llm=flaky, persist=False, approvals=store).resume(paused.run_id)
    assert again.state == "completed"
    assert calls == ["\u043d\u0430\u0439\u0434\u0438 \u0432 \u0432\u0435\u0431\u0435 bm25"]
    assert flaky.n == 2


def test_live_search_does_not_invent_results(monkeypatch):
    settings = get_settings().model_copy(
        update={"web_search_mock": False, "distill_enabled": False}
    )
    rt = AgentRuntime(db=None, llm=_llm_text(), persist=False, settings=settings)
    tool = rt.registry.get("web_search")
    assert tool is not None

    def boom(*_a, **_k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("prism.agents.tools.builtin.safe_get", boom)
    out = tool.execute("q")
    assert out["results"] == []
    assert out["status"] == "error"
    assert out["origin"] == "live"
    assert out["ok"] is False
    assert "mock" not in out["provider"]

    def http_boom(*_a, **_k):
        raise httpx.HTTPError("503")

    monkeypatch.setattr("prism.agents.tools.builtin.safe_get", http_boom)
    http_out = tool.execute("q")
    assert http_out["status"] == "error" and http_out["results"] == []

    class _BoomSoup:
        def __init__(self, *_a, **_k) -> None:
            raise ValueError("bad html")

    def ok_html(*_a, **_k):
        return b"<html></html>", "text/html", "https://lite.duckduckgo.com/"

    monkeypatch.setattr("prism.agents.tools.builtin.safe_get", ok_html)
    monkeypatch.setattr("prism.agents.tools.builtin.BeautifulSoup", _BoomSoup)
    parsed = tool.execute("q")
    assert parsed["status"] == "error"
    assert parsed["results"] == []
    assert "parse" in parsed["error"]

    monkeypatch.setattr("prism.agents.tools.builtin.BeautifulSoup", __import__("bs4").BeautifulSoup)

    def empty(*_a, **_k):
        return b"<html><body>nothing</body></html>", "text/html", "https://lite.duckduckgo.com/"

    monkeypatch.setattr("prism.agents.tools.builtin.safe_get", empty)
    blank = tool.execute("q")
    assert blank["status"] == "empty"
    assert blank["ok"] is True
    assert blank["results"] == []
    assert blank["origin"] == "live"


def test_mock_mode_is_labeled_and_example_com_is_not_banned_by_name():
    settings = get_settings().model_copy(update={"web_search_mock": True})
    rt = AgentRuntime(db=None, llm=_llm_text(), persist=False, settings=settings)
    out = rt.registry.get("web_search").execute("bm25")  # type: ignore[union-attr]
    assert out["origin"] == "mock"
    assert out["results"][0]["url"].startswith("https://example.com/")
    kept = evidence_snippets(
        [
            {
                "title": "t",
                "url": "https://example.com/a",
                "snippet": "REAL",
                "origin": "live",
                "provider": "duckduckgo_lite",
            }
        ],
        None,
        mock_mode=False,
    )
    assert kept[0]["snippet"] == "REAL"
    dropped = evidence_snippets(
        [
            {
                "title": "t",
                "url": "https://example.com/a",
                "snippet": "FAKE",
                "origin": "mock",
                "provider": "mock_error",
            }
        ],
        None,
        mock_mode=False,
    )
    assert dropped == []


def test_live_search_failure_resume_keeps_kb_and_drops_mocks(monkeypatch):
    calls: list[int] = []

    def boom(*_a, **_k):
        calls.append(1)
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("prism.agents.tools.builtin.safe_get", boom)
    store = ApprovalStore(ttl_seconds=60)
    recorder = _Recorder()
    settings = get_settings().model_copy(
        update={"web_search_mock": False, "space_filter": "lab", "distill_enabled": False}
    )
    rt = AgentRuntime(db=None, llm=recorder, persist=False, approvals=store, settings=settings)
    rt.registry.get("search_kb").execute = _kb_hit  # type: ignore[union-attr]
    paused = rt.handle(RESEARCH)
    assert paused.state == "awaiting_approval"
    assert calls == []
    assert (
        _EPHEMERAL_RUNS[paused.run_id].result["checkpoint"]["hits"][0]["text"] == "KB-UNIQUE-ALPHA"
    )
    store.approve(store.pending()[0].id)
    rt2 = AgentRuntime(
        db=None,
        llm=recorder,
        persist=False,
        approvals=store,
        settings=get_settings().model_copy(
            update={"web_search_mock": True, "distill_enabled": False}
        ),
    )
    done = rt2.resume(paused.run_id)
    assert calls == [1]
    assert rt2.settings.space_filter == "lab"
    assert rt2.settings.web_search_mock is False
    assert done.state == "completed"
    assert any("KB-UNIQUE-ALPHA" in prompt for prompt in recorder.prompts)
    joined = "\n".join(recorder.prompts)
    assert "example.com" not in joined
    assert "Mock web snippet" not in joined
    assert "\u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u044b" in done.answer
    assert all(item.get("origin") != "mock" for item in done.citations if item.get("kind") == "web")
    assert any(item["text"] == "KB-UNIQUE-ALPHA" for item in done.citations)


def test_approved_fetch_page_still_blocks_loopback():
    store = ApprovalStore(ttl_seconds=60)
    settings = get_settings().model_copy(
        update={"web_search_mock": False, "distill_enabled": False}
    )
    rt = AgentRuntime(db=None, llm=_llm_text(), persist=False, approvals=store, settings=settings)
    tool = rt.registry.get("fetch_page")
    assert tool is not None
    run = rt._create_run(role="web", objective="fetch")
    args = {"url": "http://127.0.0.1/secret"}
    waiting = rt.gate.execute(run=run, tool=tool, args=args)
    assert waiting.outcome == "approval_required"
    assert waiting.executed is False
    store.approve(waiting.approval_id or "")
    done = rt.gate.execute(run=run, tool=tool, args=args)
    assert done.executed is True
    assert done.output["origin"] == "blocked"
    assert done.output["ok"] is False
    assert "Mock page" not in (done.output.get("text") or "")
    assert "secret" not in (done.output.get("text") or "")
