from prism.config import Settings
from prism.ingest.fingerprint import index_fingerprint, index_pipeline_spec


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
