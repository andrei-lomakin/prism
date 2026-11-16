"""Backward-compatible stub; full coverage in test_policy_gate.py."""

from prism.agents.policy import ActionGate


def test_unknown_asks():
    gate = ActionGate()
    decision = gate.evaluate(run=None, tool=object(), args={})
    assert decision.mode == "ask"
