from prism.config import Settings
from prism.llm.gateway import LLMGateway, redact_pii


def test_redact_pii():
    text = redact_pii("mail me at a.b@example.com or +1 234-567-8901")
    assert "[EMAIL]" in text
    assert "@" not in text.replace("[EMAIL]", "")


def test_cache_hit(monkeypatch):
    calls = {"n": 0}

    def fake_ollama(self, **kwargs):
        from prism.llm.gateway import LLMResult

        calls["n"] += 1
        return LLMResult(text="hi", model="m", tokens_in=1, tokens_out=1, provider="ollama")

    monkeypatch.setattr(LLMGateway, "_ollama_chat", fake_ollama)
    gw = LLMGateway(Settings(llm_cache_enabled=True, llm_pii_redact=False, llm_provider="ollama"))
    a = gw.chat(user_message="q", system_prompt="s")
    b = gw.chat(user_message="q", system_prompt="s")
    assert calls["n"] == 1
    assert a.cached is False
    assert b.cached is True
