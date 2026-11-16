"""Observability: OTel spans + in-process trace buffer (Stage 5)."""

from prism.observability.tracing import get_tracer, setup_tracing, span, trace_buffer

__all__ = ["get_tracer", "setup_tracing", "span", "trace_buffer"]
