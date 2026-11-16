"""Cross-encoder reranker."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker %s", model_name)
    return CrossEncoder(model_name)


def rerank(
    query: str,
    texts: list[str],
    top_n: int,
    *,
    model_name: str = DEFAULT_MODEL,
) -> list[tuple[int, float]]:
    """Return list of (original_index, score) sorted by score desc, length ≤ top_n."""
    if not texts:
        return []
    if top_n <= 0:
        return []
    try:
        model = _load_cross_encoder(model_name)
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs)
        ranked = sorted(
            ((i, float(s)) for i, s in enumerate(scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]
    except Exception:
        logger.exception("Reranker failed; falling back to original order")
        return [(i, 0.0) for i in range(min(top_n, len(texts)))]
