# Prism

**Diploma project** — a personal research assistant over a private knowledge base.

Prism combines a transparent **RAG** pipeline with a **multi-agent** orchestration layer. You upload documents and links, ask a question or set a research goal, and the system retrieves evidence, optionally delegates specialist agents, and returns an answer with citations. Every stage is measurable: retrieval quality, generation faithfulness, latency, and cost.

Author: **Andrei Lomakin**. License: **MIT**.

---

## Motivation

Most “chat with your docs” demos hide the retrieval stack behind opaque frameworks. For an ML diploma that is the wrong trade-off: the interesting part is *which* chunking, embedding, search, reranking, and rewriting choices actually move quality — and by how much.

Prism keeps the ML core explicit and ablatable. The web UI and API stay deliberately thin so effort stays on retrieval science, evaluation, and agent control rather than product chrome.

---

## What you get

| Layer | Role |
|---|---|
| **Ingest** | PDF, DOCX, Markdown, TXT, HTML, URL → normalized text → chunks |
| **RAG** | Pluggable chunking, embeddings, dense / lexical / hybrid search, reranker, query rewriting, grounded generation with citation checks |
| **Eval** | Golden dataset, retrieval & generation metrics, ablation reports (quality × latency × $) |
| **Agents** | Orchestrator + specialist roles (retriever, web, analyst, critic), tool policy gate, budgets, run traces |
| **Ops** | Docker Compose (API, worker, Postgres+pgvector, Redis), local LLM via Ollama or optional OpenAI-compatible APIs |

---

## Architecture (high level)

```
UI (Jinja) ──► FastAPI
                 │
                 ├─► Postgres + pgvector   (documents, chunks, vectors, runs, memory)
                 ├─► BM25 index on disk    (lexical / hybrid)
                 ├─► Redis + arq worker    (ingest, agent runs, eval jobs)
                 ├─► LLMGateway            (Ollama | OpenAI-compatible)
                 └─► Embeddings / reranker (local sentence-transformers / cross-encoder)
```

**Design rules**

1. **One interface per concern** — swap chunker / embedder / retriever without rewriting the stack.
2. **Fail-closed tool policy** — unknown or risky actions require human approval.
3. **Grounded answers** — numbered sources `[n]`; missing evidence → explicit refusal.
4. **Measure, don’t assert** — every RAG change is backed by eval numbers on a held-out set.

---

## Repository layout

```
src/prism/
  api/           # FastAPI + minimal UI
  ingest/        # parsers + indexing pipeline
  rag/           # chunking, embeddings, index, retrieve, rerank, rewrite, generate
  llm/           # LLM gateway (providers, usage hooks)
  agents/        # runtime, roles, tools, policy gate, memory
  eval/          # metrics, judge, ablation runner
  worker/        # background jobs (arq)
  db/            # SQLAlchemy models, pgvector
configs/         # RAG / eval / policy YAML
data/            # demo corpus + golden set (licenses documented)
tests/
```

---

## Current status

**Stage 0 foundation is in place:** end-to-end upload → chunk → embed → dense retrieve → grounded answer on Postgres/pgvector, simple UI, worker queue, package skeleton for agents/eval/rerank/rewrite, unit tests and CI.

Upcoming stages: golden dataset & eval runner, full RAG ablations (chunking C1–C7, hybrid search, reranker, rewriting), agent runtime with policy gate, observability, and a short demo scenario.

---

## Quick start

```bash
cp .env.example .env
docker compose up -d --build
```

Open **http://localhost:8000**

Ollama should be reachable from the API container (`OLLAMA_BASE_URL`, default `http://host.docker.internal:11434`).  
Optional: `docker compose --profile local-llm up -d` and pull a chat model (e.g. `qwen2.5`).

Local (Postgres+pgvector and Redis required):

```bash
pip install -e ".[dev]"
prism init-db
uvicorn prism.api.app:app --reload --port 8000
arq prism.worker.settings.WorkerSettings
```

---

## Tests

```bash
pytest -q
ruff check src tests
```

---

## License

MIT — see [LICENSE](LICENSE).
