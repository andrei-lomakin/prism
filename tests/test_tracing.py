from prism.observability import span, trace_buffer


def test_span_buffer_records():
    trace_buffer.clear()
    tid = trace_buffer.start_trace("t-test")
    with span("unit.span", attributes={"k": 1}):
        pass
    spans = trace_buffer.get(tid)
    assert len(spans) >= 1
    assert spans[0]["name"] == "unit.span"
    assert spans[0]["duration_ms"] is not None
