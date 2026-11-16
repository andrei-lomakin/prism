"""Hybrid fusion: RRF and weighted normalized score sum."""

from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns (id, score) sorted desc."""
    scores: dict[str, float] = {}
    weights = weights or [1.0] * len(ranked_lists)
    for w, ranking in zip(weights, ranked_lists, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def weighted_fuse(
    score_maps: list[dict[str, float]],
    *,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """Weighted sum of min-max normalized score maps.

    For two maps (dense, lexical): score = α * dense_norm + (1-α) * lex_norm.
    For >2 maps, equal mix after dense/lex pattern is not used — average.
    """
    if not score_maps:
        return []
    norms = [_minmax(m) for m in score_maps]
    keys: set[str] = set()
    for m in norms:
        keys |= set(m)
    if len(norms) == 2:
        dense, lex = norms
        fused = {
            key: alpha * dense.get(key, 0.0) + (1.0 - alpha) * lex.get(key, 0.0) for key in keys
        }
    else:
        fused = {key: sum(m.get(key, 0.0) for m in norms) / len(norms) for key in keys}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
