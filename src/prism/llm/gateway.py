"""LLMGateway: single entry for all model calls.

Stage 0: Ollama + optional OpenAI-compatible. Later: PII redaction, cache,
fallback chain, token/cost accounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from prism.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    provider: str = "ollama"


class LLMGateway:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def chat(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        provider = self.settings.llm_provider
        if provider == "ollama":
            return self._ollama_chat(
                user_message=user_message,
                system_prompt=system_prompt,
                model=model or self.settings.llm_model,
                temperature=temperature,
            )
        if provider == "openai_compatible":
            return self._openai_chat(
                user_message=user_message,
                system_prompt=system_prompt,
                model=model or self.settings.openai_model or self.settings.llm_model,
                temperature=temperature,
            )
        raise ValueError(f"Unknown llm_provider: {provider}")

    def list_models(self) -> list[dict[str, Any]]:
        if self.settings.llm_provider != "ollama":
            return []
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/tags"
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
        out = []
        for m in payload.get("models") or []:
            name = m.get("name") or m.get("model")
            if name:
                out.append({"name": name, "size": m.get("size")})
        return sorted(out, key=lambda x: x["name"])

    def _ollama_chat(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
    ) -> LLMResult:
        import time

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        body = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = client.post(url, json=body)
            if resp.status_code >= 400:
                detail = resp.text
                try:
                    detail = resp.json().get("error") or detail
                except Exception:
                    pass
                raise RuntimeError(f"Ollama {resp.status_code}: {detail}")
            data = resp.json()
        ms = int((time.perf_counter() - t0) * 1000)
        content = ((data.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError(f"Empty response from Ollama: {data!r}")
        return LLMResult(
            text=content,
            model=model,
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
            latency_ms=ms,
            provider="ollama",
        )

    def _openai_chat(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
    ) -> LLMResult:
        import time

        base = (self.settings.openai_api_base or "").rstrip("/")
        if not base or not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_BASE and OPENAI_API_KEY required")
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = client.post(f"{base}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        ms = int((time.perf_counter() - t0) * 1000)
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        return LLMResult(
            text=content,
            model=model,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            latency_ms=ms,
            provider="openai_compatible",
        )


@lru_cache
def get_gateway() -> LLMGateway:
    return LLMGateway()
