"""Keep mock rows out of live evidence. Filtering is by origin, not by URL host."""

from __future__ import annotations

from typing import Any

_MOCK_PROVIDERS = frozenset({"mock", "mock_error", "mock_fallback"})

MOCK_NOTE = "Веб-источники тестовые (offline mock), не результат настоящего поиска."
UNAVAILABLE_NOTE = "Внешние источники недоступны. Ответ только по имеющимся фрагментам базы знаний."
EMPTY_NOTE = "Внешний поиск не дал результатов. Ответ только по имеющимся фрагментам базы знаний."
INSUFFICIENT_NOTE = (
    "Недостаточно источников: внешние данные недоступны, а в базе нет фрагментов по запросу."
)


def item_origin(item: dict[str, Any]) -> str:
    origin = item.get("origin")
    if origin in ("mock", "live", "blocked"):
        return str(origin)
    provider = str(item.get("provider") or "")
    if provider in _MOCK_PROVIDERS or provider.startswith("mock"):
        return "mock"
    return "live"


def evidence_snippets(
    results: list[dict[str, Any]] | None,
    pages: list[dict[str, Any]] | None,
    *,
    mock_mode: bool,
) -> list[dict[str, Any]]:
    """Page text replaces the search snippet for the same URL. Mock rows drop in live mode."""
    by_url: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def consider(item: dict[str, Any], *, prefer_text: bool) -> None:
        provider = str(item.get("provider") or "")
        if provider in {"mock_error", "mock_fallback"}:
            return
        origin = item_origin(item)
        if origin == "blocked" or item.get("ok") is False:
            return
        if origin == "mock" and not mock_mode:
            return
        url = str(item.get("url") or "").strip()
        text = ""
        if prefer_text:
            text = str(item.get("text") or item.get("snippet") or "")
        else:
            text = str(item.get("snippet") or item.get("text") or "")
        if not text and not url:
            return
        key = url or f"row-{len(order)}"
        row = {
            "title": item.get("title") or url or "web",
            "url": url,
            "snippet": text,
            "text": text,
            "origin": origin,
            "provider": item.get("provider"),
        }
        if key not in by_url:
            order.append(key)
            by_url[key] = row
        elif prefer_text and text:
            by_url[key] = row

    for item in results or []:
        if isinstance(item, dict):
            consider(item, prefer_text=False)
    for item in pages or []:
        if isinstance(item, dict):
            consider(item, prefer_text=True)
    return [by_url[key] for key in order]


def user_web_note(
    *,
    status: str | None,
    unavailable: bool,
    snippets: list[dict[str, Any]],
    has_kb: bool,
    web_attempted: bool,
) -> str | None:
    if any(item.get("origin") == "mock" for item in snippets):
        return MOCK_NOTE
    if not web_attempted:
        return None
    failed = unavailable or status in ("error", "empty")
    if not failed or snippets:
        return None
    if not has_kb:
        return INSUFFICIENT_NOTE
    if status == "empty":
        return EMPTY_NOTE
    return UNAVAILABLE_NOTE
