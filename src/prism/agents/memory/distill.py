"""S14: distill an AgentRun into a journal memory note."""

from __future__ import annotations

import re
from typing import Any

from prism.agents.memory import MemoryStore

_FACT = re.compile(r"(?m)^\s*[-*•]\s+(.+)$|\[(\d+)\]")


def distill_run_to_memory(
    memory: MemoryStore,
    *,
    run_id: str | int,
    objective: str,
    answer: str,
    citations: list[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact journal note; soft-supersede previous key for same run."""
    cites = citations or []
    titles = [str(c.get("title") or "") for c in cites[:5] if c.get("title")]
    bullets: list[str] = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if line.startswith(("-", "*", "•")) or re.match(r"^\d+\.", line):
            bullets.append(line.lstrip("-*• ").strip())
        if len(bullets) >= 5:
            break
    if not bullets:
        snippet = (answer or "").strip().replace("\n", " ")
        bullets = [snippet[:240]] if snippet else ["(empty answer)"]

    content = (
        f"objective: {objective.strip()}\n"
        f"plan: {(plan or {}).get('complexity', '?')} roles={(plan or {}).get('roles')}\n"
        f"facts:\n"
        + "\n".join(f"- {b}" for b in bullets)
        + ("\nources: " + ", ".join(titles) if titles else "")
    )
    # typo fix sources
    content = content.replace("\nources:", "\nsources:")
    key = f"run:{run_id}"
    mid = memory.remember("journal", key, content, meta={"run_id": str(run_id)})
    return {"id": mid, "layer": "journal", "key": key, "chars": len(content)}
