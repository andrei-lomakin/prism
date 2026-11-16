from prism.rag.embeddings.base import HashEmbedder


def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=32)
    a = e.embed_query("hello")
    b = e.embed_query("hello")
    c = e.embed_query("world")
    assert a == b
    assert a != c
    assert len(a) == 32
    # roughly unit length
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6
