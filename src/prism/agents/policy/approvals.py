"""Approval queue with TTL (S11)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


def args_fingerprint(tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class ApprovalRecord:
    id: str
    run_id: str | int | None
    tool: str
    args: dict[str, Any]
    rule_id: str
    created_at: float
    expires_at: float | None
    status: str = "pending"  # pending | approved | denied | expired
    reason: str | None = None

    def is_live(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if self.status == "approved":
            if self.expires_at is not None and now > self.expires_at:
                self.status = "expired"
                return False
            return True
        return False


@dataclass
class ApprovalStore:
    ttl_seconds: int = 300
    _items: dict[str, ApprovalRecord] = field(default_factory=dict)
    _seq: int = 0

    def request(
        self,
        *,
        run_id: str | int | None,
        tool: str,
        args: dict[str, Any],
        rule_id: str,
    ) -> ApprovalRecord:
        self._seq += 1
        fp = args_fingerprint(tool, args)
        key = f"{run_id}:{tool}:{fp}"
        existing = self._items.get(key)
        if existing and existing.status == "pending":
            return existing
        now = time.time()
        exp = now + self.ttl_seconds if self.ttl_seconds > 0 else None
        rec = ApprovalRecord(
            id=f"apr-{self._seq}",
            run_id=run_id,
            tool=tool,
            args=args,
            rule_id=rule_id,
            created_at=now,
            expires_at=exp,
            status="pending",
        )
        self._items[key] = rec
        return rec

    def approve(self, approval_id: str, *, reason: str | None = None) -> ApprovalRecord | None:
        rec = self._by_id(approval_id)
        if rec is None:
            return None
        rec.status = "approved"
        rec.reason = reason
        if self.ttl_seconds > 0:
            rec.expires_at = time.time() + self.ttl_seconds
        return rec

    def deny(self, approval_id: str, *, reason: str | None = None) -> ApprovalRecord | None:
        rec = self._by_id(approval_id)
        if rec is None:
            return None
        rec.status = "denied"
        rec.reason = reason
        return rec

    def is_approved(self, *, run_id: str | int | None, tool: str, args: dict[str, Any]) -> bool:
        key = f"{run_id}:{tool}:{args_fingerprint(tool, args)}"
        rec = self._items.get(key)
        return bool(rec and rec.is_live())

    def pending(self) -> list[ApprovalRecord]:
        now = time.time()
        out: list[ApprovalRecord] = []
        for rec in self._items.values():
            if rec.status == "pending":
                if rec.expires_at is not None and now > rec.expires_at:
                    rec.status = "expired"
                    continue
                out.append(rec)
        return sorted(out, key=lambda r: r.created_at)

    def _by_id(self, approval_id: str) -> ApprovalRecord | None:
        for rec in self._items.values():
            if rec.id == approval_id:
                return rec
        return None


# Process-wide store (API + agents share in single process)
_GLOBAL: ApprovalStore | None = None


def get_approval_store(ttl_seconds: int | None = None) -> ApprovalStore:
    global _GLOBAL
    if _GLOBAL is None:
        from prism.config import get_settings

        ttl = ttl_seconds if ttl_seconds is not None else get_settings().approval_ttl_seconds
        _GLOBAL = ApprovalStore(ttl_seconds=ttl)
    elif ttl_seconds is not None:
        _GLOBAL.ttl_seconds = ttl_seconds
    return _GLOBAL
