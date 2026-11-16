"""S16: structured agent reply contract + tolerant JSON parser."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

AgentActionKind = Literal["continue", "tool", "done", "ask", "delegate"]


class AgentReply(BaseModel):
    reply: str = ""
    action: AgentActionKind | None = "done"
    slots: dict[str, Any] = Field(default_factory=dict)
    tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def _extract_json_blob(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    m = _FENCE.search(raw)
    if m:
        return m.group(1).strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    m2 = _OBJ.search(raw)
    return m2.group(0) if m2 else None


def parse_agent_reply(text: str, *, default_reply: str | None = None) -> AgentReply:
    """Tolerant parse: JSON object, fenced block, or plain text → done."""
    blob = _extract_json_blob(text)
    if blob:
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                # aliases
                if "message" in data and "reply" not in data:
                    data["reply"] = data.pop("message")
                if "args" in data and "tool_args" not in data:
                    data["tool_args"] = data.pop("args")
                return AgentReply.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass
    return AgentReply(
        reply=(default_reply if default_reply is not None else (text or "").strip()),
        action="done",
        slots={},
    )


def parse_agent_reply_with_retry(
    text: str,
    *,
    repair_fn: Any | None = None,
    max_retries: int = 1,
) -> tuple[AgentReply, int]:
    """Try parse; optionally call repair_fn(raw, error) → new text."""
    reply = parse_agent_reply(text)
    # Heuristic: if plain text and looks like JSON fail already handled —
    # only retry when blob existed but validation failed via repair_fn
    blob = _extract_json_blob(text)
    if blob is None or reply.action is not None:
        # If we got structured fields from JSON we're good; if plain text also ok
        if blob is None:
            return reply, 0
        try:
            AgentReply.model_validate(json.loads(blob))
            return reply, 0
        except Exception as exc:
            if repair_fn is None or max_retries <= 0:
                return reply, 0
            repaired = repair_fn(text, str(exc))
            return parse_agent_reply(repaired), 1
    return reply, 0
