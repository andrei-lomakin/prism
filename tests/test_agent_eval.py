from prism.eval.agent_eval import run_agent_scenarios


def test_agent_scenarios_mock():
    summary = run_agent_scenarios("data/agent_scenarios.jsonl")
    assert summary["n"] == 30
    assert summary["failed"] == 0
    assert summary["success_rate"] == 1.0
