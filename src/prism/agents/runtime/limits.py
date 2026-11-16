"""Run budgets + read-call dedup + forced-synthesis flag."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunBudget:
    max_steps: int = 12
    max_searches: int = 4
    max_tokens: int = 8000
    max_cost_usd: float = 0.5
    max_active_seconds: float = 120.0


@dataclass
class RunUsage:
    steps: int = 0
    searches: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    read_fingerprints: set[str] = field(default_factory=set)
    skipped_dedup: int = 0
    forced_synthesis: bool = False
    partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "searches": self.searches,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "skipped_dedup": self.skipped_dedup,
            "forced_synthesis": self.forced_synthesis,
            "partial": self.partial,
        }


def fingerprint_call(tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LimitTracker:
    def __init__(self, budget: RunBudget | None = None) -> None:
        self.budget = budget or RunBudget()
        self.usage = RunUsage()

    def would_exceed(self) -> bool:
        u, b = self.usage, self.budget
        return (
            u.steps >= b.max_steps
            or u.searches >= b.max_searches
            or u.tokens >= b.max_tokens
            or u.cost_usd >= b.max_cost_usd
        )

    def note_step(self) -> None:
        self.usage.steps += 1

    def note_search(self) -> None:
        self.usage.searches += 1

    def note_llm(self, tokens: int, cost: float) -> None:
        self.usage.tokens += tokens
        self.usage.cost_usd += cost

    def is_duplicate_read(self, tool: str, args: dict[str, Any], *, side_effect: str) -> bool:
        if side_effect not in ("read", "none"):
            return False
        fp = fingerprint_call(tool, args)
        if fp in self.usage.read_fingerprints:
            self.usage.skipped_dedup += 1
            return True
        self.usage.read_fingerprints.add(fp)
        return False

    def mark_forced(self) -> None:
        self.usage.forced_synthesis = True
        self.usage.partial = True
