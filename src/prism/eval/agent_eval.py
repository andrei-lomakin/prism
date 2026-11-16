"""Agent scenario eval (Stage 5): success / steps / gate blocks / forced synth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism.agents.runtime import AgentRuntime, RunBudget
from prism.eval.dataset import resolve_data_path
from prism.llm.gateway import LLMResult

ROOT = Path(__file__).resolve().parents[3]


def _mock_llm(*, user_message: str, system_prompt: str) -> LLMResult:
    return LLMResult(
        text="Сводка по контексту [1].",
        model="mock",
        tokens_in=12,
        tokens_out=8,
        cost_usd=0.0,
        latency_ms=1,
    )


def run_agent_scenarios(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = resolve_data_path(path) if Path(path).exists() else ROOT / path
        if not p.exists():
            p = ROOT / path
    scenarios: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    results: list[dict[str, Any]] = []
    ok = 0
    forced = 0
    blocked = 0
    steps_sum = 0

    for sc in scenarios:
        budget = RunBudget(
            max_steps=int(sc["max_steps"]) if "max_steps" in sc else 12,
            max_searches=int(sc["max_searches"]) if "max_searches" in sc else 4,
        )
        rt = AgentRuntime(
            db=None,
            llm=_mock_llm,
            budget=budget,
            persist=False,
            # Offline scenarios auto-approve ask tools (web/fetch) when requested
            approval_fn=(lambda *_a, **_k: True) if sc.get("auto_approve", True) else None,
        )
        ans = rt.handle(sc["objective"], role=sc.get("role") or "orchestrator")
        if sc.get("expect_state"):
            success = ans.state == sc["expect_state"]
        elif sc.get("expect_partial"):
            success = ans.state == "completed" and ans.partial
        else:
            success = ans.state == "completed"
        if sc.get("expect_complexity"):
            success = success and ans.plan.get("complexity") == sc["expect_complexity"]
        gate_asks = sum(1 for e in ans.audit if e.get("decision") == "ask")
        if ans.partial:
            forced += 1
        blocked += gate_asks
        steps_sum += len(ans.steps)
        if success:
            ok += 1
        results.append(
            {
                "id": sc.get("id"),
                "success": success,
                "state": ans.state,
                "partial": ans.partial,
                "steps": len(ans.steps),
                "gate_asks": gate_asks,
                "complexity": ans.plan.get("complexity"),
            }
        )

    n = len(scenarios) or 1
    multi = [r for r in results if r.get("complexity") in ("multi", "research")]
    simple = [r for r in results if r.get("complexity") == "simple"]

    def _avg_steps(rows: list[dict]) -> float:
        return (sum(r["steps"] for r in rows) / len(rows)) if rows else 0.0

    summary = {
        "n": len(scenarios),
        "passed": ok,
        "failed": len(scenarios) - ok,
        "success_rate": ok / n,
        "avg_steps": steps_sum / n,
        "forced_synthesis_rate": forced / n,
        "gate_ask_events": blocked,
        "single_vs_multi": {
            "simple_n": len(simple),
            "simple_avg_steps": round(_avg_steps(simple), 3),
            "simple_success": sum(1 for r in simple if r["success"]) / max(1, len(simple)),
            "multi_n": len(multi),
            "multi_avg_steps": round(_avg_steps(multi), 3),
            "multi_success": sum(1 for r in multi if r["success"]) / max(1, len(multi)),
        },
        "results": results,
    }
    out = ROOT / "reports" / "AGENT_EVAL_LATEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["report"] = str(out)
    return summary
