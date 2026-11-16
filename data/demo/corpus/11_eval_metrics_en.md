# Evaluation Metrics for RAG

## Retrieval Metrics
Recall@k, Precision@k, MRR@10, nDCG@10, and Hit rate measure whether relevant documents or passages appear high in the ranking. Implement them yourself and unit-test on toy rankings.

## Generation Metrics
Faithfulness estimates whether answer claims are supported by context. Answer relevance checks topical fit. Citation accuracy requires that `[n]` points to a chunk that actually supports the claim. Unanswerable questions should trigger an explicit refusal.

## LLM-as-Judge
A fixed judge model and prompt score generation quality. Validate the judge against ≥50 human labels and report agreement (for example Cohen's κ).

## Ablations
Change one factor at a time: chunking → embedder → search → reranker → rewriting. Report quality, p50/p95 latency, and cost per 1000 queries.
