from prism.agents.memory import MemoryStore
from prism.agents.tools.builtin import ToolContext, build_default_registry


def test_sandbox_arith_ok():
    reg = build_default_registry(ToolContext(db=None, memory=MemoryStore(None)))
    tool = reg.get("python_sandbox")
    assert tool is not None
    out = tool.execute(code="print(2+3*4)")
    assert out["ok"] is True
    assert out["stdout"] == "14"


def test_sandbox_rejects_import():
    reg = build_default_registry(ToolContext(db=None, memory=MemoryStore(None)))
    tool = reg.get("python_sandbox")
    assert tool is not None
    out = tool.execute(code="print(__import__('os'))")
    assert out["ok"] is False
