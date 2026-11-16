"""Tool registry. Every side-effecting call must pass ActionGate first."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    code: str
    category: str
    side_effect: str  # none | read | write | external
    schema: dict[str, Any]
    execute: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.code] = tool

    def get(self, code: str) -> ToolSpec | None:
        return self._tools.get(code)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())


from prism.agents.tools.builtin import ToolContext, build_default_registry  # noqa: E402

__all__ = ["ToolSpec", "ToolRegistry", "ToolContext", "build_default_registry"]
