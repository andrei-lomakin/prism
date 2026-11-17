"""Retrieval metrics (own implementation, unit-tested)."""

from __future__ import annotations


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = relevant.intersection(retrieved[:k])
    return len(hit) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Standard Precision@k: |relevant ∩ retrieved[:k]| / k.

    Denominator is always ``k`` (requested cutoff), not ``len(retrieved[:k])``.
    Short lists are padded with implicit non-relevant slots — e.g. 1 hit when
    k=5 yields 0.2, not 1.0.
    """
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return len(relevant.intersection(top)) / k


def mrr_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    return 1.0 if relevant.intersection(retrieved[:k]) else 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    import math

    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        score += (2**rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Binary nDCG@k with gold-based IDCG.

    Ideal ranking is ``min(|relevant|, k)`` ones, not a re-sort of only the
    hits already in ``retrieved[:k]``. Missing gold docs still lower the score.
    """
    if k <= 0:
        return 0.0
    rels = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    n_ideal = min(len(relevant), k)
    ideal = [1.0] * n_ideal + [0.0] * (k - n_ideal)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def context_precision_at_k(relevance_labels: list[bool], k: int) -> float:
    """RAGAS-style context precision — not Precision@k.

    For each relevant position ``i`` in the top-``k`` context list, take
    precision@i, then average over relevant positions only::

        CP = Σ_i (precision@i · 1[relevant_i]) / Σ_i 1[relevant_i]

    Short lists are padded with non-relevant slots up to ``k``.
    Structural labels only (doc_key ± section) — not LLM semantic judge.
    """
    if k <= 0:
        return 0.0
    labels = list(relevance_labels[:k]) + [False] * max(0, k - len(relevance_labels))
    numer = 0.0
    n_rel = 0
    hits = 0
    for i, rel in enumerate(labels, start=1):
        if not rel:
            continue
        hits += 1
        n_rel += 1
        numer += hits / i
    if n_rel == 0:
        return 0.0
    return numer / n_rel


def context_recall_at_k(
    gold_units: list[tuple[str, str | None]],
    retrieved_units: list[tuple[str, str | None]],
    k: int,
) -> float:
    """Fraction of gold evidence units covered in top-k contexts.

    Each gold ``(doc_key, section?)`` is one unit. A hit matches when doc_key
    equals and gold section soft-matches ``section_path`` (substring either
    way; missing gold section → doc-level match). Differs from Recall@k, which
    ignores section.
    """
    if not gold_units or k <= 0:
        return 0.0
    top = retrieved_units[:k]
    matched = 0
    for g_key, g_sec in gold_units:
        for h_key, h_sec in top:
            if h_key == g_key and section_soft_match(g_sec, h_sec):
                matched += 1
                break
    return matched / len(gold_units)


def section_soft_match(gold_section: str | None, section_path: str | None) -> bool:
    """Soft section check after doc_key already matched.

    Missing ``section_path`` on the chunk (e.g. C2 has no headings) degrades to
    doc-level match — cannot refute what the indexer did not store.
    """
    if not gold_section:
        return True
    if not section_path:
        return True
    g = gold_section.lower().strip()
    p = section_path.lower().strip()
    return g == p or g in p or p in g


def chunk_relevance_labels(
    gold_units: list[tuple[str, str | None]],
    retrieved_units: list[tuple[str, str | None]],
) -> list[bool]:
    """Per-chunk label: True if chunk matches any gold unit (section-aware)."""
    return [
        any(h_key == g_key and section_soft_match(g_sec, h_sec) for g_key, g_sec in gold_units)
        for h_key, h_sec in retrieved_units
    ]


def answer_correctness_lexical(gold: str, answer: str) -> float:
    """Token F1 between gold answer and model answer (0..1)."""
    import re

    def toks(s: str) -> set[str]:
        return set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", (s or "").lower()))

    g, a = toks(gold), toks(answer)
    if not g and not a:
        return 1.0
    if not g or not a:
        return 0.0
    inter = len(g & a)
    prec = inter / len(a)
    rec = inter / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)
