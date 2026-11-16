import time

from prism.agents.policy import ActionGate, load_policy
from prism.agents.policy.approvals import ApprovalStore
from prism.agents.tools import ToolSpec


def test_approval_ttl_expires():
    store = ApprovalStore(ttl_seconds=1)
    rec = store.request(run_id=1, tool="web_search", args={"query": "x"}, rule_id="r")
    store.approve(rec.id)
    assert store.is_approved(run_id=1, tool="web_search", args={"query": "x"})
    time.sleep(1.1)
    assert store.is_approved(run_id=1, tool="web_search", args={"query": "x"}) is False


def test_gate_queues_approval_id():
    store = ApprovalStore(ttl_seconds=60)
    gate = ActionGate(load_policy(), approvals=store)
    tool = ToolSpec(
        code="web_search",
        category="external",
        side_effect="external",
        schema={},
        execute=lambda query: {"ok": True},
    )
    result = gate.execute(run=type("R", (), {"id": 7})(), tool=tool, args={"query": "q"})
    assert result.executed is False
    assert result.approval_id
    store.approve(result.approval_id)
    result2 = gate.execute(run=type("R", (), {"id": 7})(), tool=tool, args={"query": "q"})
    assert result2.executed is True
