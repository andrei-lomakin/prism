"""Generate additional MIT educational notes to grow the demo corpus."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo" / "corpus"

TOPICS = [
    ("dense_vs_sparse", "Dense vs sparse retrieval", "Плотный и разреженный поиск"),
    ("rrf_math", "Reciprocal Rank Fusion math", "Математика RRF"),
    ("hnsw", "HNSW indexes", "Индексы HNSW"),
    ("ivfflat", "IVFFlat indexes", "Индексы IVFFlat"),
    ("tokenizers", "Tokenizers for RAG", "Токенизаторы для RAG"),
    ("packing", "Context packing budgets", "Бюджеты упаковки контекста"),
    ("hallucination", "Hallucination failure modes", "Режимы галлюцинаций"),
    ("citation_formats", "Citation formats", "Форматы цитирования"),
    ("latency_cost", "Latency vs cost tradeoffs", "Латентность и стоимость"),
    ("cold_start", "Cold start corpora", "Холодный старт корпусов"),
    ("multilingual", "Multilingual retrieval", "Мультиязычный поиск"),
    ("query_expansion", "Query expansion", "Расширение запроса"),
    ("hyde_notes", "HyDE notes", "Заметки о HyDE"),
    ("cross_encoder", "Cross-encoder rerankers", "Cross-encoder реранкеры"),
    ("agent_budgets", "Agent budgets", "Бюджеты агентов"),
    ("tool_policies", "Tool policies", "Политики инструментов"),
    ("observability", "Observability for RAG", "Наблюдаемость RAG"),
    ("evaluation_pitfalls", "Evaluation pitfalls", "Ловушки оценки"),
    ("data_licenses", "Data licenses for demos", "Лицензии данных для демо"),
    ("security_pii", "PII in LLM gateways", "PII в LLM-шлюзах"),
]


def _body_en(title: str, slug: str) -> str:
    return f"""# {title}

## Definition

Educational note on **{title}** for the Prism diploma corpus (`{slug}`).

## Why it matters

Retrieval quality depends on indexing choices, ranking fusion, and evaluation
discipline. This note states the core idea without proprietary content.

## Practical tips

- Prefer measurable ablations over anecdotes.
- Keep chunk keys stable for golden sets.
- Log fallbacks and latency separately from quality metrics.

## Summary

`{title}` is a building block of transparent RAG systems.
"""


def _body_ru(title: str, slug: str) -> str:
    return f"""# {title}

## Определение

Учебная заметка про **{title}** для демо-корпуса Prism (`{slug}`).

## Зачем это важно

Качество поиска зависит от индексов, слияния рангов и дисциплины оценки.
Текст намеренно нейтральный и без проприетарного содержимого.

## Практические советы

- Сравнивайте конфигурации на golden set.
- Не привязывайте разметку к `chunk_id`.
- Фиксируйте fallback и латентность отдельно от качества.

## Кратко

`{title}` — базовый блок прозрачного RAG.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = 23
    n = 0
    for i, (slug, en, ru) in enumerate(TOPICS):
        num_en = start + i * 2
        num_ru = num_en + 1
        p_en = OUT / f"{num_en:02d}_{slug}_en.md"
        p_ru = OUT / f"{num_ru:02d}_{slug}_ru.md"
        if not p_en.exists():
            p_en.write_text(_body_en(en, slug), encoding="utf-8")
            n += 1
        if not p_ru.exists():
            p_ru.write_text(_body_ru(ru, slug), encoding="utf-8")
            n += 1
    print(
        f"wrote_or_skipped topics={len(TOPICS)} new_files~={n} "
        f"total={len(list(OUT.glob('*.md')))}"
    )


if __name__ == "__main__":
    main()
