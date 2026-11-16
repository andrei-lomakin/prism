from prism.rag.retrieve.fusion import rrf_fuse, weighted_fuse


def test_rrf_prefers_shared_top():
    a = ["d1", "d2", "d3"]
    b = ["d2", "d1", "d4"]
    fused = rrf_fuse([a, b], k=60)
    ids = [x[0] for x in fused]
    assert ids[0] in {"d1", "d2"}


def test_weighted_alpha():
    dense = {"a": 1.0, "b": 0.2}
    lex = {"a": 0.1, "b": 1.0}
    high_dense = weighted_fuse([dense, lex], alpha=0.9)
    high_lex = weighted_fuse([dense, lex], alpha=0.1)
    assert high_dense[0][0] == "a"
    assert high_lex[0][0] == "b"
