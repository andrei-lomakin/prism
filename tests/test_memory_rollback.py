from prism.agents.memory import MemoryStore


def test_rollback_reactivates_previous():
    mem = MemoryStore(None)
    a = mem.remember("journal", "k", "v1")
    b = mem.remember("journal", "k", "v2")
    assert mem.recall("k")[0].content == "v2"
    assert mem.rollback(a)["ok"] is True
    assert mem.recall("k")[0].content == "v1"
    hist = mem.history("journal", "k")
    assert len(hist) == 2
    assert b != a
