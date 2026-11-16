# Retrieval-Augmented Generation Overview

## Definition
Retrieval-Augmented Generation (RAG) combines a retriever with a generative language model. The retriever finds relevant passages from a knowledge base; the generator conditions its answer on those passages.

## Why RAG
Parametric LLM knowledge is static after training and may hallucinate. RAG injects up-to-date or private documents at inference time, which improves factual grounding when citations are enforced.

## Typical Pipeline
1. Ingest documents and split them into chunks.
2. Embed chunks into a vector index.
3. At query time, embed the query and retrieve top-k chunks.
4. Pack chunks into a prompt and generate an answer with citations.

## Failure Modes
Common failures include bad chunking boundaries, weak embeddings for the corpus language, retrieving the wrong document, and answers that cite sources they did not use.
