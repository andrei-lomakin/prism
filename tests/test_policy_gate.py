"""ActionGate: unknown→ask, locked floor, sole execute path."""

from __future__ import annotations

from pathlib import Path

from prism.agents.policy import ActionGate, apply_overlay, load_policy, merge_rules, redact_args
from prism.agents.tools import ToolSpec


def test_unknown_asks():
    gate = ActionGate(merge_rules({"system": [], "workspace": [], "agent": []}))
    decision = gate.evaluate(run=None, tool=object(), args={})
    assert decision.mode == "ask"
    assert decision.rule_id == "default.unknown"


def test_known_auto_from_default_yaml():
    gate = ActionGate(load_policy())
    tool = ToolSpec(
        code="search_kb",
        category="retrieval",
        side_effect="read",
        schema={},
        execute=lambda query: {"ok": True, "query": query},
    )
    decision = gate.evaluate(run=None, tool=tool, args={"query": "x"})
    assert decision.mode == "auto"
    result = gate.execute(run=None, tool=tool, args={"query": "x"})
    assert result.executed is True
    assert result.output["ok"] is True


def test_ask_blocks_without_approval():
    gate = ActionGate(load_policy())
    tool = ToolSpec(
        code="web_search",
        category="external",
        side_effect="external",
        schema={},
        execute=lambda query: {"hits": [query]},
    )
    result = gate.execute(run=None, tool=tool, args={"query": "leak"})
    assert result.executed is False
    assert result.decision.mode == "ask"


def test_ask_runs_with_approval_fn():
    gate = ActionGate(load_policy(), approval_fn=lambda *_a, **_k: True)
    tool = ToolSpec(
        code="web_search",
        category="external",
        side_effect="external",
        schema={},
        execute=lambda query: {"hits": [query]},
    )
    result = gate.execute(run=None, tool=tool, args={"query": "ok"})
    assert result.executed is True


def test_locked_rule_cannot_weaken():
    # Locked ask cannot be weakened to auto via agent overlay.
    locked = merge_rules(
        {
            "system": [
                {"id": "sys.web", "tool": "web_search", "mode": "ask", "locked": True},
            ],
            "workspace": [],
            "agent": [],
        }
    )
    new_b, ok, reason = apply_overlay(
        locked, tool="web_search", mode="auto", rule_id="agent.hack", layer="agent"
    )
    assert ok is False
    assert "cannot be weakened" in reason
    assert new_b.rule_for("web_search").mode == "ask"


def test_redact_secrets():
    red = redact_args({"api_key": "sk-secret", "query": "hello", "token": "abc"})
    assert red["api_key"] == "***"
    assert red["token"] == "***"
    assert red["query"] == "hello"


def test_sole_execute_path_no_direct_tool_execute_outside_gate():
    """Grep-style: agent role/runtime modules must not call tool.execute directly."""
    root = Path(__file__).resolve().parents[1] / "src" / "prism" / "agents"
    offenders: list[str] = []
    allowed = {
        root / "policy" / "gate.py",
        # ToolSpec stores execute callable; builtin defines it — OK
        root / "tools" / "builtin.py",
        root / "tools" / "__init__.py",
    }
    for path in root.rglob("*.py"):
        if path in allowed or "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if "tool.execute" in line or ".execute(**" in line:
                # roles/runtime must go through gate.execute
                if "gate.execute" in line:
                    continue
                offenders.append(f"{path.relative_to(root.parent.parent)}:{i}:{line.strip()}")
    assert offenders == [], "direct tool.execute outside ActionGate:\n" + "\n".join(offenders)
