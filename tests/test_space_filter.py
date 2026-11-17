from unittest.mock import patch

from prism.config import Settings
from prism.rag.retrieve.service import (
    _lexical_search,
    dense_search_filters,
    retrieve,
)
from prism.rag.retrieve.spaces import apply_space_weights
from prism.rag.retrieve.types import RetrievedChunk


def _hit(i: int, space: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        document_id=i,
        document_name=f"d{i}",
        text="x",
        score=score,
        space=space,
    )


def test_space_weights_prefer_personal():
    hits = [_hit(1, "shared", 1.0), _hit(2, "personal", 0.9)]
    out = apply_space_weights(hits, {"personal": 1.0, "project": 0.85, "shared": 0.72})
    assert out[0].space == "personal"


def test_hard_filter_space():
    hits = [_hit(1, "shared"), _hit(2, "personal"), _hit(3, "shared")]
    filtered = [h for h in hits if h.space == "shared"]
    assert len(filtered) == 2


def test_dense_filters_include_space_before_limit():
    clauses = dense_search_filters(model_name="m", space="project")
    rendered = str(clauses[-1].compile(compile_kwargs={"literal_binds": True}))
    assert "project" in rendered
    assert "space" in rendered
    assert len(dense_search_filters(model_name="m", space=None)) == 3


def _search_settings(**extra) -> Settings:
    base = dict(
        retrieval_mode="dense",
        top_k=1,
        rewrite_strategy="R0",
        rerank_enabled=False,
        rerank_candidates=5,
        space_weights_enabled=False,
    )
    base.update(extra)
    return Settings(**base)


def test_retrieve_asks_dense_for_space_before_cutoff():
    """Other spaces fill the global top-k; the project hit exists only inside the filter."""
    calls: list[str | None] = []

    def fake_dense(db, query, *, candidate_k, settings, space=None):
        calls.append(space)
        if space == "project":
            return [_hit(9, "project", score=0.2)]
        return [_hit(i, "personal", score=0.99) for i in range(1, 5)]

    for rerank in (False, True):
        calls.clear()
        settings = _search_settings(rerank_enabled=rerank, rerank_candidates=5)
        with (
            patch("prism.rag.retrieve.service._dense_search", side_effect=fake_dense),
            patch(
                "prism.rag.retrieve.service.apply_cross_encoder_rerank",
                side_effect=lambda hits, query, *, top_n, model_name, rerank_fn=None: hits[:top_n],
            ),
        ):
            hits, _mode = retrieve(
                None, "q", top_k=1, settings=settings, space="project", pack=False
            )
        assert calls == ["project"]
        assert [h.chunk_id for h in hits] == [9]


def test_lexical_keeps_in_space_hit_below_global_cutoff():
    ranked = [(str(i), float(10 - i)) for i in (1, 2, 3, 4, 5, 9)]
    captured: dict[str, int] = {}

    class Store:
        def __init__(self, path, collection="default"):
            self.path = path

        def corpus_size(self):
            return len(ranked)

        def search(self, query, top_k=10):
            captured["k"] = top_k
            return ranked[:top_k]

    spaces = {1: "personal", 2: "shared", 3: "personal", 4: "shared", 5: "personal", 9: "project"}
    rows = [
        {
            "id": i,
            "document_id": i,
            "text": f"t{i}",
            "section_path": None,
            "page": None,
            "space": spaces[i],
            "chunk_meta": {},
            "document_name": f"d{i}",
            "document_meta": {},
        }
        for i in spaces
    ]

    class DB:
        def execute(self, stmt):
            class Result:
                def mappings(self):
                    return rows

            return Result()

    with patch("prism.rag.retrieve.service.BM25Store", Store):
        hits = _lexical_search(
            DB(),
            "q",
            candidate_k=2,
            settings=_search_settings(retrieval_mode="lexical"),
            space="project",
        )
    assert captured["k"] == len(ranked)
    assert [h.chunk_id for h in hits] == [9]
