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
| **Eval** | Golden dataset, retrieval & generation metrics, ablation reports (quality × latency × $), git provenance in every report |
| **Agents** | Orchestrator + Retriever / Web / Analyst / Critic, ActionGate, budgets, distill, traces |
| **Ops** | Docker Compose (API, worker, Postgres+pgvector, Redis), Ollama or OpenAI-compatible APIs |

---

## Current status

**Stages 0–7 complete** (diploma slice per TZ). Honest gaps: Cohen's κ (human second pass), GIF not committed, sync subagents, corpus still << TZ 300–1000 (expandable via `just expand-corpus`).

Defense pack: [`docs/DEFENSE.md`](docs/DEFENSE.md) · [`docs/slides.md`](docs/slides.md) · [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md) · [`docs/adr/`](docs/adr/)

---

## Quick start

```bash
cp .env.example .env
docker compose up -d --build
pip install -e ".[dev]"   # or: just install
just demo                 # init-db + ingest-demo + rebuild-bm25
```

Open **http://127.0.0.1:8000** (prefer `127.0.0.1` on Windows).

Full walkthrough: [`docs/DEMO.md`](docs/DEMO.md).

---

## Reproduce (defense)

```bash
just check              # fmt + lint + pytest
just defense-prep       # check + provenance + ablation table + agent eval
# with DB up + indexed corpus:
just defense-test-eval  # held-out hybrid + dense on test.jsonl
```

Every eval report under `reports/<run_id>/` includes:

- `metrics.json` / `summary.md` / `per_question.jsonl`
- **`provenance.commit`** (git SHA, branch, dirty flag)

```bash
prism eval provenance
prism eval compare      # → reports/ABLATION_TABLE.md + ABLATION_RECALL.svg
prism eval kappa --a data/golden/annotator_a.example.jsonl --b data/golden/annotator_b.example.jsonl
```

Canonical numbers: [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md).

Optional auth: set `AUTH_PASSWORD` in `.env` (HTTP Basic on `/api/*`; `/api/health` stays open).

---

## Architecture (high level)

```
UI (Jinja) ──► FastAPI
                 │
                 ├─► Postgres + pgvector   (documents, chunks, vectors, runs, memory)
                 ├─► BM25 index on disk    (lexical / hybrid)
                 ├─► Redis + arq worker    (ingest jobs)
                 ├─► LLMGateway            (Ollama | OpenAI-compatible)
                 └─► Embeddings / reranker (local sentence-transformers / cross-encoder)
```

Design rules: one interface per concern · fail-closed tool policy · grounded `[n]` answers · measure, don’t assert.

Deep dive (Russian, gitignored in some clones): `docs/architecture.md`.

---

## Eval snapshot (dev, hybrid RRF)

| Metric | mean |
|---|---:|
| Recall@5 | 0.934 |
| MRR@10 | 0.754 |
| nDCG@10 | 0.817 |
| Hit@5 | 1.000 |

Default chat model: `qwen3:4b` (Ollama). Embedder: `paraphrase-multilingual-MiniLM-L12-v2`. Postgres host port **5434**.

---

## Lint / format / test

Requires [just](https://github.com/casey/just).

```bash
just check / just fix / just test / just lint / just fmt
```

---

## Slides

Marp source: `docs/slides.md` → prebuilt `docs/slides.pdf`.

```bash
npx --yes @marp-team/marp-cli docs/slides.md -o docs/slides.pdf
```

---

## Attribution / third-party

| Component | License / note |
|---|---|
| Demo corpus (`data/demo/corpus/`) | Original MIT educational notes for this diploma |
| `sentence-transformers` / MiniLM | Apache-2.0 (Hugging Face model cards) |
| `bm25s`, FastAPI, SQLAlchemy, pgvector | respective OSS licenses |
| Ollama models (e.g. qwen3) | follow model card terms |

No scraped copyrighted books/articles in the committed corpus.

## License

MIT — see [LICENSE](LICENSE).
