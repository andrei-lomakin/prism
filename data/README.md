# Data

## Demo corpus (`demo/corpus/`)

Authored educational notes on RAG / embeddings / hybrid search / agents / eval (RU + EN).

| Field | Value |
|---|---|
| Documents | 22+ Markdown files (grow with `just expand-corpus`) |
| License | MIT (same as the Prism repository) — original text written for this diploma |
| Purpose | Reproducible indexing + golden questions without third-party scrape risk |

Filename stem = stable **`doc_key`** used by the golden set (survives re-chunking).

Index:

```bash
just expand-corpus   # optional extra educational notes
prism ingest-demo
```

## Golden set (`golden/`)

| File | Role |
|---|---|
| `dev.jsonl` | tuning / ablations |
| `test.jsonl` | held-out; do **not** tune hyperparameters here |
| `stats.json` | distribution summary |
| `ANNOTATION.md` | labeling protocol + κ plan |

Build / rebuild:

```bash
prism golden build --target 240
```

Schema fields: `id`, `question`, `language`, `type`, `relevant_docs[{doc_key, section}]`, `gold_answer`, `split`, `reviewed`, `notes`.

## Other

- `uploads/` — user uploads (gitignored content)
- `bm25/` — lexical indexes (stage 3)
