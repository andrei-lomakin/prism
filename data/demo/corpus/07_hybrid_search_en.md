# Dense, Lexical, and Hybrid Search

## Dense Search
Dense search uses vector similarity. It captures paraphrases but may miss rare identifiers, error codes, and exact jargon.

## Lexical Search (BM25)
BM25 scores term overlap with document length normalization. It is strong on exact keywords and complements dense retrieval.

## Hybrid Fusion
Reciprocal Rank Fusion (RRF) with k=60 merges ranked lists without calibrated score scales. Alternatively, a weighted sum of normalized dense and lexical scores uses α ∈ {0.5, 0.72, 0.75, 0.9}.

## Fallbacks
If the embedder times out, fall back to lexical-only, then to simple keyword matching. Record the retrieval mode in traces and metrics.

## Context Packing
After retrieval, pack at most N chunks, respect a token budget, enforce diversity (for example ≤3 chunks per document), and drop scores below a threshold.
