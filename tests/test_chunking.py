from prism.rag.chunking import chunk_text


def test_c1_word_window_respects_size():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, strategy="C1", max_chars=40, overlap=10)
    assert chunks
    assert all(len(c.text) <= 50 for c in chunks)


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
    vocab = set(text.split())
    for c in chunks:
        for w in c.text.split():
            assert w in vocab


def test_c4_recursive():
    text = "A" * 30 + "\n\n" + "B" * 30 + ". Next sentence here."
    chunks = chunk_text(text, strategy="C4", max_chars=40, overlap=5)
    assert chunks
    # overlap can grow a chunk slightly past max_chars
    assert all(len(c.text) <= 80 for c in chunks)


def test_c5_structure_section_path():
    text = "# Title\n\n## Section A\n\nHello world paragraph.\n\n## Section B\n\nOther text here."
    chunks = chunk_text(text, strategy="C5", max_chars=200, overlap=0)
    assert chunks
    assert any(c.section_path and "Section" in c.section_path for c in chunks)


def test_c6_semantic():
    text = (
        "Cats like milk. Cats sleep a lot. "
        "Quantum chromodynamics studies quarks. Gluons mediate the strong force."
    )
    chunks = chunk_text(text, strategy="C6", max_chars=500, overlap=0)
    assert len(chunks) >= 1


def test_c7_parent_child_meta():
    text = "First long enough paragraph about retrieval methods and embeddings.\n\nSecond paragraph about hybrid search and BM25 lexical matching."
    chunks = chunk_text(text, strategy="C7", max_chars=80, overlap=10)
    assert chunks
    assert all(c.meta.get("strategy") == "C7" for c in chunks)
    assert any(c.meta.get("parent_text") for c in chunks)
