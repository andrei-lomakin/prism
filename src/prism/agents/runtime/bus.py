"""S17: simplified in-process message bus (urgent wakes waiter)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BusMessage:
    id: int
    channel: str
    sender: str
    body: str
    urgent: bool = False
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    def __init__(self) -> None:
        self._seq = 0
        self._channels: dict[str, list[BusMessage]] = defaultdict(list)
        self._events: dict[str, threading.Event] = defaultdict(threading.Event)
        self._lock = threading.Lock()

    def post(
        self,
        channel: str,
        body: str,
        *,
        sender: str = "system",
        urgent: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> BusMessage:
        with self._lock:
            self._seq += 1
            msg = BusMessage(
                id=self._seq,
                channel=channel,
                sender=sender,
                body=body,
                urgent=urgent,
                meta=meta or {},
            )
            self._channels[channel].append(msg)
            if urgent:
                self._events[channel].set()
            return msg

    def read(self, channel: str, *, after_id: int = 0, limit: int = 50) -> list[BusMessage]:
        with self._lock:
            msgs = [m for m in self._channels.get(channel, []) if m.id > after_id]
            return msgs[:limit]

    def wait(self, channel: str, *, timeout: float = 0.0) -> bool:
        """Block until urgent message or timeout. timeout=0 → poll once."""
        ev = self._events[channel]
        if timeout <= 0:
            return ev.is_set()
        ok = ev.wait(timeout=timeout)
        if ok:
            ev.clear()
        return ok


_BUS: MessageBus | None = None


def get_bus() -> MessageBus:
    global _BUS
    if _BUS is None:
        _BUS = MessageBus()
    return _BUS
