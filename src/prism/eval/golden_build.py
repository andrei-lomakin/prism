"""Build Stage 1 golden JSONL from curated templates.

Why templates (not live LLM generation) for the seed set:
- reproducible without API cost;
- every item is tied to an authored corpus section;
- easy to explain on defense.

Expansion to 200+ items uses controlled paraphrase patterns per seed fact.
Manual review flag marks the primary hand-checked subset (≥30).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from prism.eval.schema import GoldenQuestion, RelevantDoc

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "golden"


def _rd(*pairs: tuple[str, str | None]) -> list[RelevantDoc]:
    return [RelevantDoc(doc_key=k, section=s) for k, s in pairs]


# Seed facts: (id_prefix, lang, type, question, answer, docs, reviewed, notes)
def _seeds() -> list[dict]:
    s: list[dict] = []

    def add(**kw):
        s.append(kw)

    # --- factoid EN/RU ---
    add(
        key="f_rag_def_en",
        language="en",
        type="factoid",
        question="What is Retrieval-Augmented Generation?",
        gold_answer="RAG combines a retriever over a knowledge base with a generative LLM that conditions on retrieved passages.",
        docs=_rd(("01_rag_overview_en", "Definition")),
        reviewed=True,
    )
    add(
        key="f_rag_def_ru",
        language="ru",
        type="factoid",
        question="Что такое Retrieval-Augmented Generation (RAG)?",
        gold_answer="RAG соединяет поисковик по базе знаний и генеративную модель, которая отвечает с опорой на найденные фрагменты.",
        docs=_rd(("02_rag_overview_ru", "Определение")),
        reviewed=True,
    )
    add(
        key="f_pipeline_en",
        language="en",
        type="factoid",
        question="List the four steps of a typical RAG pipeline.",
        gold_answer="Ingest and chunk; embed into a vector index; retrieve top-k at query time; pack context and generate with citations.",
        docs=_rd(("01_rag_overview_en", "Typical Pipeline")),
        reviewed=True,
    )
    add(
        key="f_overlap_en",
        language="en",
        type="factoid",
        question="How should chunk overlap treat word boundaries?",
        gold_answer="Overlap must not split words; boundaries should snap to whitespace or tokenizer tokens.",
        docs=_rd(("03_chunking_en", "Fixed Windows")),
        reviewed=True,
    )
    add(
        key="f_overlap_ru",
        language="ru",
        type="factoid",
        question="Как overlap при чанкинге должен относиться к границам слов?",
        gold_answer="Overlap не должен резать слово: граница ставится по пробелу или токену токенизатора.",
        docs=_rd(("04_chunking_ru", "Фиксированные окна")),
        reviewed=True,
    )
    add(
        key="f_mix_forbid_en",
        language="en",
        type="factoid",
        question="Can vectors from different embedding models share one index?",
        gold_answer="No. Mixing models is forbidden; changing the model requires a full re-index and embedding_model must be stored per chunk.",
        docs=_rd(("05_embeddings_en", "Mixing Models is Forbidden")),
        reviewed=True,
    )
    add(
        key="f_e5_prefix_en",
        language="en",
        type="factoid",
        question="Which prefixes do E5 embedding models expect?",
        gold_answer="E5 models expect query: and passage: prefixes.",
        docs=_rd(("05_embeddings_en", "Model Families")),
        reviewed=True,
    )
    add(
        key="f_bm25_en",
        language="en",
        type="factoid",
        question="What does BM25 optimize for compared to dense retrieval?",
        gold_answer="BM25 scores term overlap with length normalization and is strong on exact keywords, complementing dense paraphrase matching.",
        docs=_rd(("07_hybrid_search_en", "Lexical Search (BM25)")),
        reviewed=True,
    )
    add(
        key="f_rrf_en",
        language="en",
        type="factoid",
        question="What is the default RRF constant k used in Prism hybrid search?",
        gold_answer="k = 60.",
        docs=_rd(("19_numeric_baselines_en", "RRF Constant")),
        reviewed=True,
    )
    add(
        key="f_gate_en",
        language="en",
        type="factoid",
        question="What happens when an agent tool is unknown to the policy gate?",
        gold_answer="Unknown tools fail closed to ask mode.",
        docs=_rd(("13_agents_en", "Tool Policy Gate")),
        reviewed=True,
    )
    add(
        key="f_gate_ru",
        language="ru",
        type="factoid",
        question="Что делает policy-гейт при неизвестном инструменте?",
        gold_answer="Неизвестный инструмент переводится в режим ask (fail closed).",
        docs=_rd(("14_agents_ru", "Policy-гейт инструментов")),
        reviewed=True,
    )
    add(
        key="f_pgvector_en",
        language="en",
        type="factoid",
        question="Which SQL command enables pgvector in PostgreSQL?",
        gold_answer="CREATE EXTENSION IF NOT EXISTS vector.",
        docs=_rd(("15_pgvector_en", "Operational Notes")),
        reviewed=True,
    )
    add(
        key="f_refuse_en",
        language="en",
        type="factoid",
        question="What should a grounded RAG system do when evidence is missing?",
        gold_answer="It must refuse rather than invent an answer.",
        docs=_rd(("17_grounding_en", "Numbered Sources")),
        reviewed=True,
    )

    # --- numeric ---
    add(
        key="n_alpha_en",
        language="en",
        type="numeric",
        question="Which dense weight α values are used as hybrid ablation starting points?",
        gold_answer="0.5, 0.72, 0.75, and 0.9.",
        docs=_rd(("19_numeric_baselines_en", "Fusion Weights")),
        reviewed=True,
    )
    add(
        key="n_alpha_ru",
        language="ru",
        type="numeric",
        question="Какие стартовые значения α для гибридного поиска перечислены в корпусе?",
        gold_answer="0.5, 0.72, 0.75 и 0.9.",
        docs=_rd(("20_numeric_baselines_ru", "Веса слияния")),
        reviewed=True,
    )
    add(
        key="n_k_rerank_en",
        language="en",
        type="numeric",
        question="Which candidate pool sizes K are used before cross-encoder reranking?",
        gold_answer="K ∈ {20, 50}.",
        docs=_rd(("19_numeric_baselines_en", "Candidate Widths")),
        reviewed=True,
    )
    add(
        key="n_chunk_default_en",
        language="en",
        type="numeric",
        question="What are the Stage 0 default chunk_size and overlap?",
        gold_answer="chunk_size = 256 approximate tokens, overlap = 32 (~12.5%).",
        docs=_rd(("19_numeric_baselines_en", "Chunk Sizes")),
        reviewed=True,
    )
    add(
        key="n_topn_en",
        language="en",
        type="numeric",
        question="Which top-n values are kept after reranking in the ablation grid?",
        gold_answer="n ∈ {3, 5, 8}.",
        docs=_rd(("09_rerank_rewrite_en", "Cross-Encoder Reranking")),
        reviewed=True,
    )

    # --- comparison ---
    add(
        key="c_dense_bm25_en",
        language="en",
        type="comparison",
        question="Compare dense retrieval and BM25: what is each strong at?",
        gold_answer="Dense captures paraphrases; BM25 is stronger on exact keywords and rare identifiers. Hybrid combines both.",
        docs=_rd(
            ("07_hybrid_search_en", "Dense Search"),
            ("08_hybrid_search_ru", "Лексический поиск (BM25)"),
        ),
        reviewed=True,
    )
    add(
        key="c_chunk_strats_en",
        language="en",
        type="comparison",
        question="Compare fixed windows versus paragraph packing for chunking.",
        gold_answer="Fixed windows are simple sliding budgets; paragraph packing preserves linguistic boundaries better than naive character cuts.",
        docs=_rd(
            ("03_chunking_en", "Fixed Windows"),
            ("04_chunking_ru", "Упаковка абзацев и предложений"),
        ),
        reviewed=True,
    )
    add(
        key="c_parent_child_en",
        language="en",
        type="comparison",
        question="What is the difference between child and parent chunks in small-to-big retrieval?",
        gold_answer="Children are small for precise retrieval; parents are larger and returned to the generator for broader context.",
        docs=_rd(("03_chunking_en", "Parent-Child (Small-to-Big)")),
        reviewed=True,
    )
    add(
        key="c_minilm_en",
        language="en",
        type="comparison",
        question="Why might multilingual MiniLM beat English-only MiniLM on a Russian corpus?",
        gold_answer="English-only MiniLM is weak on RU text; multilingual models usually improve Recall@k on Russian passages.",
        docs=_rd(("05_embeddings_en", "Language Mismatch")),
        reviewed=True,
    )
    add(
        key="c_r0_r2_en",
        language="en",
        type="comparison",
        question="When is R0 preferable to multi-query rewriting R2?",
        gold_answer="On clean single-shot factoids R0 is often competitive and cheaper; R2 helps vague or multi-turn queries.",
        docs=_rd(("09_rerank_rewrite_en", "When Rewriting Helps")),
        reviewed=True,
    )
    add(
        key="c_spaces_ru",
        language="ru",
        type="comparison",
        question="Чем отличаются пространства personal, project и shared при поиске?",
        gold_answer="Это разные уровни знаний; веса между ними — гиперпараметры абляции, а не фиксированные продуктовые правила.",
        docs=_rd(("22_spaces_memory_ru", "Пространства")),
        reviewed=True,
    )

    # --- multi-hop ---
    add(
        key="m_rag_metrics_en",
        language="en",
        type="multi-hop",
        question="How does a typical RAG pipeline connect to retrieval metrics like Recall@k?",
        gold_answer="The pipeline retrieves top-k chunks; Recall@k and related metrics evaluate whether the relevant evidence appears in that ranking before generation.",
        docs=_rd(
            ("01_rag_overview_en", "Typical Pipeline"), ("11_eval_metrics_en", "Retrieval Metrics")
        ),
        reviewed=True,
    )
    add(
        key="m_hybrid_packer_en",
        language="en",
        type="multi-hop",
        question="After hybrid fusion, what packing constraints are applied before generation?",
        gold_answer="At most N chunks, a token budget, diversity limits per document, and a minimum score threshold.",
        docs=_rd(
            ("07_hybrid_search_en", "Hybrid Fusion"),
            ("08_hybrid_search_ru", "Упаковка контекста"),
        ),
        reviewed=True,
    )
    add(
        key="m_embed_pg_en",
        language="en",
        type="multi-hop",
        question="If you change the embedding model, what must happen in a pgvector deployment?",
        gold_answer="Do not mix geometries: re-index fully, keep embedding_model on chunks, and migrate the vector column if dimensionality changes.",
        docs=_rd(
            ("05_embeddings_en", "Mixing Models is Forbidden"),
            ("15_pgvector_en", "Operational Notes"),
        ),
        reviewed=True,
    )
    add(
        key="m_agent_budget_en",
        language="en",
        type="multi-hop",
        question="How do agent budgets interact with forced synthesis?",
        gold_answer="Runs limit steps, tokens, cost, and active time; on exhaustion the system forces a partial synthesis instead of failing silently.",
        docs=_rd(("13_agents_en", "Budgets"), ("14_agents_ru", "Машина состояний рана")),
        reviewed=True,
    )
    add(
        key="m_ground_eval_en",
        language="en",
        type="multi-hop",
        question="How do citation checks relate to faithfulness evaluation?",
        gold_answer="Citation checks verify [n] targets; faithfulness estimates whether claims are supported by context — both fight hallucinations and are reported in metrics.",
        docs=_rd(
            ("17_grounding_en", "Post-Generation Checks"),
            ("11_eval_metrics_en", "Generation Metrics"),
        ),
        reviewed=True,
    )
    add(
        key="m_rag_chunk_ru",
        language="ru",
        type="multi-hop",
        question="Как связаны ошибки чанкинга и качество RAG-ответа?",
        gold_answer="Плохие границы чанков — типичный failure mode RAG; стратегии вроде packing/structure-aware как раз уменьшают разрыв контекста перед генерацией.",
        docs=_rd(
            ("02_rag_overview_ru", "Типичные ошибки"),
            ("04_chunking_ru", "Упаковка абзацев и предложений"),
        ),
        reviewed=True,
    )
    add(
        key="m_rrf_alpha_en",
        language="en",
        type="multi-hop",
        question="Name one rank-fusion method and one weighted-score alternative used in hybrid search ablations.",
        gold_answer="RRF with k=60, and a weighted sum of normalized scores with α in {0.5, 0.72, 0.75, 0.9}.",
        docs=_rd(
            ("07_hybrid_search_en", "Hybrid Fusion"), ("19_numeric_baselines_en", "Fusion Weights")
        ),
        reviewed=True,
    )

    # --- unanswerable ---
    add(
        key="u_weather_en",
        language="en",
        type="unanswerable",
        question="What will the weather in Minsk be tomorrow according to the knowledge base?",
        gold_answer="The knowledge base has no answer to this question.",
        docs=[],
        reviewed=True,
    )
    add(
        key="u_ceo_en",
        language="en",
        type="unanswerable",
        question="Who is the CEO of Prism according to the corpus?",
        gold_answer="The knowledge base has no answer to this question.",
        docs=[],
        reviewed=True,
    )
    add(
        key="u_gpu_price_ru",
        language="ru",
        type="unanswerable",
        question="Сколько стоит аренда A100 в час по данным этой базы?",
        gold_answer="В базе нет ответа на этот вопрос.",
        docs=[],
        reviewed=True,
    )
    add(
        key="u_kafka_en",
        language="en",
        type="unanswerable",
        question="How do I configure Kafka consumer lag alerts in Prism?",
        gold_answer="The knowledge base has no answer to this question.",
        docs=[],
        reviewed=True,
    )
    add(
        key="u_react_ru",
        language="ru",
        type="unanswerable",
        question="Какой хук React использовать для подписки на WebSocket в UI Prism?",
        gold_answer="В базе нет ответа на этот вопрос.",
        docs=[],
        reviewed=True,
    )

    return s


_PARAPHRASE_EN = [
    "In one sentence: {q}",
    "Briefly explain: {q}",
    "According to the docs, {q}",
    "From the knowledge base: {q}",
]
_PARAPHRASE_RU = [
    "Кратко: {q}",
    "По материалам базы: {q}",
    "Одним абзацем ответь: {q}",
    "Сформулируй ответ: {q}",
]


def expand_seeds(
    seeds: list[dict], *, target: int = 240, rng: random.Random | None = None
) -> list[GoldenQuestion]:
    """Expand seeds with paraphrases while keeping type quotas.

    Quotas (TZ Stage 1): unanswerable ≥10%, multi-hop ≥15%.
    """
    rng = rng or random.Random(42)
    items: list[GoldenQuestion] = []
    for seed in seeds:
        q = GoldenQuestion(
            id=f"seed_{seed['key']}",
            question=seed["question"],
            language=seed["language"],
            type=seed["type"],
            relevant_docs=seed["docs"],
            gold_answer=seed["gold_answer"],
            split="dev",  # temporary; assigned later
            reviewed=bool(seed.get("reviewed", False)),
            notes=seed.get("notes", "primary seed"),
        )
        q.validate_consistency()
        items.append(q)

    by_type: dict[str, list[dict]] = {}
    for seed in seeds:
        by_type.setdefault(seed["type"], []).append(seed)

    min_unans = max(1, int(target * 0.12))
    min_multi = max(1, int(target * 0.18))
    para_i = 0

    def add_para(seed: dict) -> None:
        nonlocal para_i
        templates = _PARAPHRASE_EN if seed["language"] == "en" else _PARAPHRASE_RU
        tmpl = rng.choice(templates)
        qtext = seed["question"]
        question = tmpl.format(q=qtext[0].lower() + qtext[1:])
        pq = GoldenQuestion(
            id=f"para_{seed['key']}_{para_i:03d}",
            question=question,
            language=seed["language"],
            type=seed["type"],
            relevant_docs=list(seed["docs"]),
            gold_answer=seed["gold_answer"],
            split="dev",
            reviewed=False,
            notes="controlled paraphrase of seed; pending secondary review",
        )
        pq.validate_consistency()
        items.append(pq)
        para_i += 1

    # Fill quota types first
    while sum(1 for x in items if x.type == "unanswerable") < min_unans:
        add_para(rng.choice(by_type["unanswerable"]))
    while sum(1 for x in items if x.type == "multi-hop") < min_multi:
        add_para(rng.choice(by_type["multi-hop"]))

    # Fill remaining with mixed paraphrases
    while len(items) < target:
        add_para(rng.choice(seeds))

    return items[:target]


def assign_splits(
    items: list[GoldenQuestion], *, test_ratio: float = 0.3, rng: random.Random | None = None
) -> None:
    """Stratified-ish split by type; hyperparameters must not be tuned on test."""
    rng = rng or random.Random(42)
    by_type: dict[str, list[GoldenQuestion]] = {}
    for it in items:
        by_type.setdefault(it.type, []).append(it)
    for group in by_type.values():
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_ratio))
        for i, it in enumerate(group):
            it.split = "test" if i < n_test else "dev"


def write_jsonl(items: list[GoldenQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def build(target: int = 240) -> dict:
    seeds = _seeds()
    items = expand_seeds(seeds, target=target)
    assign_splits(items)
    for it in items:
        it.validate_consistency()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = [i for i in items if i.split == "dev"]
    test = [i for i in items if i.split == "test"]
    write_jsonl(dev, OUT_DIR / "dev.jsonl")
    write_jsonl(test, OUT_DIR / "test.jsonl")

    def stats(xs: list[GoldenQuestion]) -> dict:
        from collections import Counter

        return {
            "n": len(xs),
            "by_type": dict(Counter(x.type for x in xs)),
            "by_lang": dict(Counter(x.language for x in xs)),
            "reviewed": sum(1 for x in xs if x.reviewed),
            "unanswerable_pct": round(
                100 * sum(1 for x in xs if x.type == "unanswerable") / max(1, len(xs)), 1
            ),
            "multihop_pct": round(
                100 * sum(1 for x in xs if x.type == "multi-hop") / max(1, len(xs)), 1
            ),
        }

    summary = {
        "total": len(items),
        "seeds": len(seeds),
        "dev": stats(dev),
        "test": stats(test),
        "all": stats(items),
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
