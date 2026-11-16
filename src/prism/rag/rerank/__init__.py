"""Cross-encoder reranker (stage 3)."""


def rerank(query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
    raise NotImplementedError("Reranker lands in stage 3")
