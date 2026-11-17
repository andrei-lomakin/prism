"""LLMGateway: Ollama + OpenAI-compatible with PII redact, cache, cost (S7)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from prism.config import Settings, get_settings
from prism.observability import span

logger = logging.getLogger(__name__)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b(?:\+?\d[\d\- ()]{8,}\d)\b")
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def redact_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _CARD.sub("[CARD]", text)
    return text


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    provider: str = "ollama"
    cached: bool = False


class _LRUCache:
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, LLMResult] = OrderedDict()

    def get(self, key: str) -> LLMResult | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        hit = self._data[key]
        return LLMResult(
            text=hit.text,
            model=hit.model,
            tokens_in=hit.tokens_in,
            tokens_out=hit.tokens_out,
            cost_usd=0.0,
            latency_ms=0,
            provider=hit.provider,
            cached=True,
        )

    def put(self, key: str, value: LLMResult) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


# Process-wide prompt cache (Variant B lite). Key already includes
# provider|model|temperature|system|user, so per-request LLMGateway(cfg)
# still hits. Not Redis — dies with the worker/API process.
_SHARED_CACHE = _LRUCache(maxsize=256)


def clear_llm_cache() -> None:
    """Drop in-process LLM responses (call after model/provider change)."""
    _SHARED_CACHE.clear()
    get_gateway.cache_clear()


class LLMGateway:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache = _SHARED_CACHE

    def chat(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        cfg = self.settings
        user = redact_pii(user_message) if cfg.llm_pii_redact else user_message
        system = redact_pii(system_prompt) if cfg.llm_pii_redact else system_prompt
        model_name = model or (
            cfg.openai_model or cfg.llm_model
            if cfg.llm_provider == "openai_compatible"
            else cfg.llm_model
        )
        cache_key = None
        if cfg.llm_cache_enabled:
            cache_key = hashlib.sha256(
                f"{cfg.llm_provider}|{model_name}|{temperature}|{system}|{user}".encode()
            ).hexdigest()
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        fallbacks = [m.strip() for m in (cfg.llm_fallback_models or "").split(",") if m.strip()]
        candidates = [model_name] + [m for m in fallbacks if m != model_name]
        last_err: Exception | None = None
        result: LLMResult | None = None
        for cand in candidates:
            try:
                with span(
                    "llm.chat",
                    attributes={
                        "provider": cfg.llm_provider,
                        "model": cand,
                        "temperature": temperature,
                    },
                ):
                    if cfg.llm_provider == "ollama":
                        result = self._ollama_chat(
                            user_message=user,
                            system_prompt=system,
                            model=cand,
                            temperature=temperature,
                        )
                    elif cfg.llm_provider == "openai_compatible":
                        result = self._openai_chat(
                            user_message=user,
                            system_prompt=system,
                            model=cand,
                            temperature=temperature,
                        )
                    else:
                        raise ValueError(f"Unknown llm_provider: {cfg.llm_provider}")
                break
            except Exception as exc:  # noqa: BLE001 — try next fallback
                last_err = exc
                logger.warning("LLM %s failed: %s", cand, exc)
                continue
        if result is None:
            raise RuntimeError(f"All LLM candidates failed: {candidates}") from last_err

        result.cost_usd = self._cost(result.tokens_in, result.tokens_out)
        if cache_key is not None:
            self._cache.put(cache_key, result)
        return result

    def chat_stream(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Generator[str, None, LLMResult]:
        """Yield real provider token/text deltas; ``return`` final ``LLMResult``.

        Cache hit → one delta with the full text (still cheaper than a round-trip).
        """
        cfg = self.settings
        user = redact_pii(user_message) if cfg.llm_pii_redact else user_message
        system = redact_pii(system_prompt) if cfg.llm_pii_redact else system_prompt
        model_name = model or (
            cfg.openai_model or cfg.llm_model
            if cfg.llm_provider == "openai_compatible"
            else cfg.llm_model
        )
        cache_key = None
        if cfg.llm_cache_enabled:
            cache_key = hashlib.sha256(
                f"{cfg.llm_provider}|{model_name}|{temperature}|{system}|{user}".encode()
            ).hexdigest()
            cached = self._cache.get(cache_key)
            if cached is not None:
                if cached.text:
                    yield cached.text
                return cached

        fallbacks = [m.strip() for m in (cfg.llm_fallback_models or "").split(",") if m.strip()]
        candidates = [model_name] + [m for m in fallbacks if m != model_name]
        last_err: Exception | None = None
        for cand in candidates:
            try:
                with span(
                    "llm.chat_stream",
                    attributes={
                        "provider": cfg.llm_provider,
                        "model": cand,
                        "temperature": temperature,
                        "stream": True,
                    },
                ):
                    if cfg.llm_provider == "ollama":
                        stream = self._ollama_chat_stream(
                            user_message=user,
                            system_prompt=system,
                            model=cand,
                            temperature=temperature,
                        )
                    elif cfg.llm_provider == "openai_compatible":
                        stream = self._openai_chat_stream(
                            user_message=user,
                            system_prompt=system,
                            model=cand,
                            temperature=temperature,
                        )
                    else:
                        raise ValueError(f"Unknown llm_provider: {cfg.llm_provider}")
                    result = yield from stream
                result.cost_usd = self._cost(result.tokens_in, result.tokens_out)
                if cache_key is not None:
                    self._cache.put(cache_key, result)
                return result
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("LLM stream %s failed: %s", cand, exc)
                continue
        raise RuntimeError(f"All LLM stream candidates failed: {candidates}") from last_err

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        cfg = self.settings
        return (tokens_in / 1_000_000.0) * cfg.llm_input_usd_per_1m + (
            tokens_out / 1_000_000.0
        ) * cfg.llm_output_usd_per_1m

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

    def _ollama_chat_stream(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
    ) -> Generator[str, None, LLMResult]:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        body = {
            "model": model,
            "stream": True,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        t0 = time.perf_counter()
        parts: list[str] = []
        tokens_in = 0
        tokens_out = 0
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            with client.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    detail = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Ollama {resp.status_code}: {detail}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(f"Ollama stream error: {data['error']}")
                    piece = (data.get("message") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        yield piece
                    if data.get("done"):
                        tokens_in = int(data.get("prompt_eval_count") or 0)
                        tokens_out = int(data.get("eval_count") or 0)
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("Empty streamed response from Ollama")
        return LLMResult(
            text=text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            provider="ollama",
        )

    def _openai_chat_stream(
        self,
        *,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
    ) -> Generator[str, None, LLMResult]:
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
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        t0 = time.perf_counter()
        parts: list[str] = []
        tokens_in = 0
        tokens_out = 0
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            with client.stream(
                "POST", f"{base}/chat/completions", headers=headers, json=body
            ) as resp:
                if resp.status_code >= 400:
                    detail = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"OpenAI {resp.status_code}: {detail}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    data = json.loads(line)
                    usage = data.get("usage") or {}
                    if usage:
                        tokens_in = int(usage.get("prompt_tokens") or tokens_in)
                        tokens_out = int(usage.get("completion_tokens") or tokens_out)
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        parts.append(delta)
                        yield delta
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("Empty streamed response from OpenAI-compatible API")
        return LLMResult(
            text=text,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            provider="openai_compatible",
        )


@lru_cache
def get_gateway() -> LLMGateway:
    """Process singleton with env defaults. Prefer ``LLMGateway(cfg)`` when
    request settings differ — prompt cache is shared either way.
    """
    return LLMGateway()


def gateway_for(settings: Settings | None = None) -> LLMGateway:
    """Build a gateway bound to ``settings``; responses use ``_SHARED_CACHE``."""
    return LLMGateway(settings)
