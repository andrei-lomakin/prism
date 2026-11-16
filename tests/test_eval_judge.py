from prism.config import Settings
from prism.eval.judge import judge_answer


def test_heuristic_unanswerable_refusal():
    s = Settings()
    scores = judge_answer(
        question="Weather?",
        answer="В базе нет ответа на этот вопрос.",
        contexts=[],
        qtype="unanswerable",
        settings=s,
        use_llm=False,
    )
    assert scores.refusal_correct == 1.0
    assert scores.method == "heuristic"


def test_heuristic_answerable():
    s = Settings()
    ctx = ["RAG combines a retriever with a generative language model."]
    scores = judge_answer(
        question="What is RAG?",
        answer="RAG combines a retriever with a generative language model [1].",
        contexts=ctx,
        qtype="factoid",
        settings=s,
        use_llm=False,
    )
    assert scores.citation_ok
    assert scores.faithfulness > 0.3
