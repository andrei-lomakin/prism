from prism.eval.relevance import gold_doc_keys, retrieved_doc_keys
from prism.eval.schema import GoldenQuestion, RelevantDoc
from prism.rag.retrieve.types import RetrievedChunk


def test_gold_and_retrieved_keys():
    q = GoldenQuestion(
        id="t",
        question="What is RAG briefly?",
        language="en",
        type="multi-hop",
        relevant_docs=[
            RelevantDoc(doc_key="01_rag_overview_en"),
            RelevantDoc(doc_key="11_eval_metrics_en"),
        ],
        gold_answer="x",
        split="dev",
    )
    assert gold_doc_keys(q) == {"01_rag_overview_en", "11_eval_metrics_en"}

    hits = [
        RetrievedChunk(1, 1, "a.md", "t", 0.9, doc_key="01_rag_overview_en"),
        RetrievedChunk(2, 1, "a.md", "t", 0.8, doc_key="01_rag_overview_en"),
        RetrievedChunk(3, 2, "b.md", "t", 0.7, doc_key="11_eval_metrics_en"),
    ]
    assert retrieved_doc_keys(hits) == ["01_rag_overview_en", "11_eval_metrics_en"]
