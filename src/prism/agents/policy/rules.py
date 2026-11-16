"""Policy rule loading + merge with locked floor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Mode = Literal["auto", "on_request", "ask", "handoff"]
LAYER_ORDER = ("system", "workspace", "agent")

# Weaker → stronger ranking for "cannot weaken locked"
_MODE_STRENGTH = {"auto": 0, "on_request": 1, "ask": 2, "handoff": 3}


@dataclass
class PolicyRule:
    id: str
    tool: str
    mode: Mode
    locked: bool = False
    layer: str = "system"


@dataclass
class PolicyBundle:
    version: int = 1
    default_mode: Mode = "ask"
    rules_by_tool: dict[str, PolicyRule] = field(default_factory=dict)
    all_rules: list[PolicyRule] = field(default_factory=list)

    def rule_for(self, tool: str) -> PolicyRule | None:
        return self.rules_by_tool.get(tool)


def _as_mode(value: Any) -> Mode:
    m = str(value or "ask")
    if m not in _MODE_STRENGTH:
        return "ask"
    return m  # type: ignore[return-value]


def _weaken(existing: Mode, proposed: Mode) -> bool:
    """True if proposed is less restrictive than existing."""
    return _MODE_STRENGTH[proposed] < _MODE_STRENGTH[existing]


def merge_rules(
    layers: dict[str, list[dict[str, Any]]], *, default_mode: Mode = "ask"
) -> PolicyBundle:
    """Merge system → workspace → agent. Locked rules reject weakening overlays."""
    bundle = PolicyBundle(default_mode=default_mode)
    by_tool: dict[str, PolicyRule] = {}

    for layer in LAYER_ORDER:
        for raw in layers.get(layer) or []:
            tool = str(raw.get("tool") or "").strip()
            if not tool:
                continue
            rule = PolicyRule(
                id=str(raw.get("id") or f"{layer}.{tool}"),
                tool=tool,
                mode=_as_mode(raw.get("mode")),
                locked=bool(raw.get("locked", False)),
                layer=layer,
            )
            prev = by_tool.get(tool)
            if prev and prev.locked and _weaken(prev.mode, rule.mode):
                # Keep locked floor; ignore weakening overlay.
                continue
            if prev and prev.locked:
                # Locked may stay or tighten; keep locked flag from earlier layer.
                rule = PolicyRule(
                    id=rule.id,
                    tool=tool,
                    mode=rule.mode if not _weaken(prev.mode, rule.mode) else prev.mode,
                    locked=True,
                    layer=layer,
                )
            by_tool[tool] = rule

    bundle.rules_by_tool = by_tool
    bundle.all_rules = list(by_tool.values())
    return bundle


def load_policy(path: Path | str | None = None) -> PolicyBundle:
    root = Path(__file__).resolve().parents[4]
    cfg_path = Path(path) if path else root / "configs" / "policies" / "default.yaml"
    if not cfg_path.is_file():
        return PolicyBundle()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    layers = raw.get("layers")
    if not layers:
        # Flat legacy list under rules:
        flat = raw.get("rules") or []
        layers = {"system": flat, "workspace": [], "agent": []}
    bundle = merge_rules(layers, default_mode=_as_mode(raw.get("default_mode")))
    bundle.version = int(raw.get("version") or 1)
    return bundle


def apply_overlay(
    bundle: PolicyBundle,
    *,
    tool: str,
    mode: Mode,
    rule_id: str,
    layer: str = "agent",
) -> tuple[PolicyBundle, bool, str]:
    """Propose a rule change. Returns (new_bundle, accepted, reason)."""
    existing = bundle.rule_for(tool)
    if existing and existing.locked and _weaken(existing.mode, mode):
        return bundle, False, f"locked rule {existing.id} cannot be weakened to {mode}"

    layers: dict[str, list[dict[str, Any]]] = {k: [] for k in LAYER_ORDER}
    for r in bundle.all_rules:
        layers.setdefault(r.layer, []).append(
            {"id": r.id, "tool": r.tool, "mode": r.mode, "locked": r.locked}
        )
    layers.setdefault(layer, []).append(
        {"id": rule_id, "tool": tool, "mode": mode, "locked": False}
    )
    merged = merge_rules(layers, default_mode=bundle.default_mode)
    merged.version = bundle.version
    # Detect reject via unchanged mode when weaken attempted
    after = merged.rule_for(tool)
    if (
        existing
        and existing.locked
        and after
        and after.mode == existing.mode
        and _weaken(existing.mode, mode)
    ):
        return bundle, False, f"locked rule {existing.id} cannot be weakened to {mode}"
    return merged, True, "applied"
