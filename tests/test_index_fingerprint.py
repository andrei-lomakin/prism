from types import SimpleNamespace

import pytest

from prism.config import Settings
from prism.ingest.fingerprint import (
    IndexFingerprintError,
    index_fingerprint,
    index_pipeline_spec,
    require_index_fingerprint,
)


def test_fingerprint_stable_for_same_settings():
    s = Settings(
        chunking_strategy="C2",
        chunk_size=256,
        chunk_overlap=32,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    assert index_fingerprint(s) == index_fingerprint(s)


def test_fingerprint_changes_with_chunking():
    a = Settings(chunking_strategy="C2", chunk_size=256, chunk_overlap=32)
    b = Settings(chunking_strategy="C5", chunk_size=256, chunk_overlap=32)
    assert index_fingerprint(a) != index_fingerprint(b)


def test_fingerprint_changes_with_embedder():
    a = Settings(embedding_model="model-a")
    b = Settings(embedding_model="model-b")
    assert index_fingerprint(a) != index_fingerprint(b)


def test_pipeline_spec_has_required_keys():
    spec = index_pipeline_spec(Settings())
    for key in (
        "preprocessing_version",
        "chunking_strategy",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "embedding_dim",
    ):
        assert key in spec


def _doc(status: str, fp: str | None, name: str = "doc.md") -> SimpleNamespace:
    meta = {"index_fingerprint": fp} if fp else {}
    return SimpleNamespace(status=status, meta=meta, name=name, id=1)


def test_require_index_fingerprint_accepts_match():
    settings = Settings(chunking_strategy="C2", embedding_model="model-a")
    fp = index_fingerprint(settings)
    assert require_index_fingerprint([_doc("indexed", fp)], settings) == fp


def test_require_index_fingerprint_rejects_chunking_mismatch():
    built = Settings(chunking_strategy="C2", embedding_model="model-a")
    asked = built.model_copy(update={"chunking_strategy": "C5"})
    with pytest.raises(IndexFingerprintError, match="Reindex"):
        require_index_fingerprint([_doc("indexed", index_fingerprint(built))], asked)


def test_require_index_fingerprint_rejects_embedder_mismatch():
    built = Settings(chunking_strategy="C2", embedding_model="model-a")
    asked = built.model_copy(update={"embedding_model": "model-b"})
    with pytest.raises(IndexFingerprintError, match="model-b"):
        require_index_fingerprint([_doc("indexed", index_fingerprint(built))], asked)


def test_require_index_fingerprint_rejects_empty_corpus():
    with pytest.raises(IndexFingerprintError, match="no indexed"):
        require_index_fingerprint([], Settings())


def test_run_eval_aborts_before_metrics_on_fingerprint_mismatch(monkeypatch, tmp_path):
    from prism.eval.runner import _load_yaml, _settings_from_rag_yaml, run_eval

    c5 = _settings_from_rag_yaml(_load_yaml("configs/rag/c5_structure.yaml"))
    c2 = c5.model_copy(update={"chunking_strategy": "C2"})
    doc = _doc("indexed", index_fingerprint(c2), name="baseline.md")

    class Result:
        def all(self):
            return [doc]

    class DB:
        def scalars(self, stmt):
            return Result()

        def close(self):
            self.closed = True

    cfg = tmp_path / "mismatch.yaml"
    cfg.write_text(
        "name: mismatch\n"
        "dataset: data/golden/dev.jsonl\n"
        "config: configs/rag/c5_structure.yaml\n"
        "max_questions: 1\n"
        "generation: false\n"
        "retrieval_only: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("prism.eval.runner.init_db", lambda: None)
    monkeypatch.setattr("prism.eval.runner.get_session_factory", lambda: lambda: DB())

    def _no_metrics(*_a, **_k):
        raise AssertionError("metrics started")

    monkeypatch.setattr("prism.eval.runner._eval_one", _no_metrics)
    with pytest.raises(IndexFingerprintError, match="C5"):
        run_eval(str(cfg))
