"""Retrieval metrics (own implementation, unit-tested)."""

from __future__ import annotations


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = relevant.intersection(retrieved[:k])
    return len(hit) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(relevant.intersection(top)) / len(top)


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
    rels = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def context_precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant (same as precision@k)."""
    return precision_at_k(relevant, retrieved, k)


def context_recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of gold docs appearing in top-k (same as recall@k)."""
    return recall_at_k(relevant, retrieved, k)


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
