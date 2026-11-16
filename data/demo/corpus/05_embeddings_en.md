# Embedding Models for Retrieval

## Role of Embeddings
Dense retrievers map queries and passages into a shared vector space. Cosine similarity or inner product ranks candidates.

## Model Families
Common families include MiniLM (fast, 384 dimensions), multilingual MiniLM for cross-lingual corpora, E5 models that expect `query:` / `passage:` prefixes, and larger models such as BGE-M3.

## Mixing Models is Forbidden
Vectors from different embedding models must not share one index. Changing the model requires a full re-index; store `embedding_model` on every chunk.

## Language Mismatch
An English-only MiniLM is a weak baseline for a Russian corpus. Multilingual or Russian-capable models usually improve Recall@k on RU text.

## Caching
Embedding the same string twice is wasteful. A hash-keyed cache speeds indexing and evaluation without changing geometry.
