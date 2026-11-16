"""Query rewriting strategies R0–R5 (stage 3)."""


def rewrite(query: str, strategy: str = "R0", history: list[dict] | None = None) -> dict:
    if strategy == "R0":
        return {"semantic_query": query, "lexical_query": query, "queries": [query]}
    raise NotImplementedError(f"Rewrite strategy {strategy} lands in stage 3")
