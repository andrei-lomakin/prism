# Reranking and Query Rewriting

## Cross-Encoder Reranking
A bi-encoder retrieves a broad candidate set (K ∈ {20, 50}). A cross-encoder reranker scores (query, passage) pairs and keeps top-n (n ∈ {3, 5, 8}). Measure reranker latency separately from retrieval.

## Query Rewriting Variants
- R0: no rewrite.
- R1: condense the dialog history into a standalone query.
- R2: multi-query — three paraphrases fused with RRF.
- R3: HyDE — generate a hypothetical answer, embed it, retrieve by that vector.
- R4: separate semantic query (with dialog context) and lexical query (last user turn only).
- R5: intent-based retrieve planner choosing sources and budgets.

## When Rewriting Helps
Multi-turn chat, vague pronouns, and short queries benefit most. On clean single-shot factoid questions, R0 is often competitive and cheaper.
