"""OpenTelemetry setup + lightweight in-process span buffer for UI traces."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "prism_trace_id", default=None
)
_setup_done = False


@dataclass
class SpanRecord:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ms: float
    end_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.end_ms is None:
            return None
        return max(0.0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "error": self.error,
        }


class TraceBuffer:
    """Ring-ish in-memory store for recent traces (UI / tests without Phoenix)."""

    def __init__(self, max_traces: int = 200) -> None:
        self.max_traces = max_traces
        self._spans: dict[str, list[SpanRecord]] = defaultdict(list)
        self._order: list[str] = []

    def start_trace(self, trace_id: str | None = None) -> str:
        tid = trace_id or uuid.uuid4().hex[:16]
        if tid not in self._spans:
            self._order.append(tid)
            while len(self._order) > self.max_traces:
                old = self._order.pop(0)
                self._spans.pop(old, None)
        _current_trace.set(tid)
        return tid

    def current_trace_id(self) -> str | None:
        return _current_trace.get()

    def add(self, span: SpanRecord) -> None:
        self._spans[span.trace_id].append(span)

    def get(self, trace_id: str) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._spans.get(trace_id, [])]

    def clear(self) -> None:
        self._spans.clear()
        self._order.clear()


trace_buffer = TraceBuffer()

_span_stack: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "prism_span_stack", default=None
)


def setup_tracing(
    service_name: str = "prism",
    *,
    enabled: bool | None = None,
    endpoint: str | None = None,
) -> None:
    """Configure OTel TracerProvider. Safe no-op if disabled or deps missing."""
    global _setup_done
    from prism.config import get_settings

    cfg = get_settings()
    on = cfg.otel_enabled if enabled is None else enabled
    if not on:
        logger.info("OTel disabled (otel_enabled=false)")
        _setup_done = True
        return
    if _setup_done:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create({"service.name": service_name or cfg.otel_service_name})
        provider = TracerProvider(resource=resource)
        ep = endpoint if endpoint is not None else cfg.otel_exporter_endpoint
        if ep:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep)))
            except Exception:
                logger.exception("OTLP exporter unavailable; using console")
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _setup_done = True
        logger.info("OTel tracing enabled service=%s endpoint=%s", service_name, ep)
    except Exception:
        logger.exception("Failed to setup OTel; continuing with buffer-only spans")
        _setup_done = True


def get_tracer(name: str = "prism"):
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return None


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[SpanRecord]:
    """Record a span into TraceBuffer and optionally OTel."""
    tid = _current_trace.get() or trace_buffer.start_trace()
    prev = _span_stack.get()
    stack = list(prev) if prev else []
    parent = stack[-1] if stack else None
    sid = uuid.uuid4().hex[:12]
    rec = SpanRecord(
        name=name,
        trace_id=tid,
        span_id=sid,
        parent_id=parent,
        start_ms=time.perf_counter() * 1000,
        attributes=dict(attributes or {}),
    )
    stack.append(sid)
    token = _span_stack.set(stack)

    otel_cm = None
    otel_span = None
    tracer = get_tracer()
    if tracer is not None:
        try:
            otel_cm = tracer.start_as_current_span(name)
            otel_span = otel_cm.__enter__()
            for k, v in (attributes or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    otel_span.set_attribute(k, v)
        except Exception:
            otel_cm = None
            otel_span = None

    try:
        yield rec
    except Exception as exc:
        rec.status = "error"
        rec.error = str(exc)
        if otel_span is not None:
            try:
                otel_span.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode

                otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
        raise
    finally:
        rec.end_ms = time.perf_counter() * 1000
        trace_buffer.add(rec)
        if otel_cm is not None:
            try:
                otel_cm.__exit__(None, None, None)
            except Exception:
                pass
        _span_stack.reset(token)
