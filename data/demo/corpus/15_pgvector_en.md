# Vector Storage with PostgreSQL and pgvector

## Why pgvector
Keeping documents, chunks, metadata, and embeddings in one database simplifies joins, transactions, and reproducible demos. Extension `vector` adds typed embedding columns and distance operators.

## Indexing
HNSW indexes accelerate approximate nearest neighbor search with cosine distance. Always filter by `embedding_model` so incompatible geometries never mix.

## Operational Notes
Schema bootstrap should run `CREATE EXTENSION IF NOT EXISTS vector`. Changing embedding dimensionality requires a migration of the vector column and a full re-index.
