from prism.rag.chunking import chunk_text


def test_c1_word_window_respects_size():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, strategy="C1", max_chars=40, overlap=10)
    assert chunks
    assert all(len(c.text) <= 50 for c in chunks)  # small slack for last words


def test_c2_paragraphs():
    text = "First paragraph.\n\nSecond paragraph is here.\n\nThird one."
    chunks = chunk_text(text, strategy="C2", max_chars=80, overlap=0)
    assert len(chunks) >= 1
    assert "First" in chunks[0].text


def test_c3_sentences():
    text = "One. Two! Three? Four."
    chunks = chunk_text(text, strategy="C3", max_chars=20, overlap=0)
    assert len(chunks) >= 2


def test_overlap_does_not_split_words():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = chunk_text(text, strategy="C1", max_chars=25, overlap=10)
    assert len(chunks) >= 2
    # every token in every chunk must be a full word from the source
    vocab = set(text.split())
    for c in chunks:
        for w in c.text.split():
            assert w in vocab


def test_c4_not_ready():
    import pytest

    with pytest.raises(NotImplementedError):
        chunk_text("x", strategy="C4", max_chars=10, overlap=0)
