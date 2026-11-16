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

**Stages 0–4:** dense + hybrid RAG (BM25, RRF/weighted), chunking C1–C7, rewrite R0–R5, optional rerank, eval + golden set, **AgentRuntime** (Orchestrator / Retriever / Critic), ActionGate policies, subagents, memory. Chat: `mode=rag|agent`.

Next: ablation table on test, observability/UI trace (Stages 5–6).

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

Index the demo corpus and (re)build the golden set:

```bash
prism ingest-demo
prism golden build --target 240
```

Eval (Stage 2):

```bash
# retrieval-only smoke (fast)
prism eval run --config configs/eval/smoke.yaml --skip-generation

# full baseline on dev (retrieve + generate + judge)
prism eval run --config configs/eval/baseline.yaml
```

Reports land in `reports/<run_id>/` (`metrics.json`, `summary.md`, `per_question.jsonl`).

Hybrid ablation example:

```bash
prism rebuild-bm25
prism eval run --config configs/eval/ablate_hybrid.yaml --skip-generation
```

UI: **http://127.0.0.1:8000** (prefer `127.0.0.1` over `localhost` on Windows/WSL).  
Postgres from host: port **5434** (see `.env.example`).

Default models: embedder `paraphrase-multilingual-MiniLM-L12-v2`, chat `qwen3:4b` via Ollama.

---

## Lint / format / test

Requires [just](https://github.com/casey/just) (`winget install just` / `cargo install just`).

```bash
just check      # fmt-check + lint + pytest
just fix        # ruff format + lint --fix
just test       # pytest -q
just lint       # ruff check
just fmt        # ruff format
```

Without just: `ruff check src tests`, `ruff format src tests`, `pytest -q`.

---

## License

MIT — see [LICENSE](LICENSE).
