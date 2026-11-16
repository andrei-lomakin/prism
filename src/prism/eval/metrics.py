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
