# Golden set annotation protocol (Stage 1)

## Goal

Produce questions whose relevance labels survive chunking changes by binding to
**`doc_key` + optional `section`**, not to `chunk_id`.

## Question types

| Type | Rule |
|---|---|
| `factoid` | One clear fact; ≥1 relevant doc/section |
| `multi-hop` | Needs ≥2 distinct `doc_key`s |
| `comparison` | Contrast two ideas; docs that contain both sides |
| `numeric` | Answer contains a number/set of numbers from the corpus |
| `unanswerable` | Not answerable from corpus; `relevant_docs` **must be empty**; gold = explicit refusal |

Languages: `ru` / `en`. Target mix: both present; unanswerable ≥10%; multi-hop ≥15%.

## Procedure

1. Read the section(s) in `data/demo/corpus/`.
2. Write the question so a competent reader can answer **only** from those sections.
3. Write `gold_answer` as a short grounded statement (or refusal).
4. Fill `relevant_docs` with precise `section` titles when possible.
5. Set `reviewed=true` only after a human pass.

## Double annotation & Cohen's κ

- Primary pass: single annotator (diploma author) on all `reviewed=true` seeds (≥30).
- Secondary pass (planned): second annotator labels the same 30 items for:
  - relevant `doc_key` set equality;
  - answerable vs unanswerable;
  - type tag.
- Report Cohen's κ on those 30 after the second pass. Until then `stats.json` records `reviewed` count; κ is **pending** (honest gap, not fabricated).
- Tooling: write two JSONL files `{id,label}` and run
  `prism eval kappa --a annotator_a.jsonl --b annotator_b.jsonl`
  (see `data/golden/annotator_*.example.jsonl` for format).

## Split policy

- Stratified by `type` into `dev` / `test` (~70/30).
- **Never** tune chunking / embedder / hybrid hyperparameters on `test`.

## Regeneration

```bash
prism golden build --target 240
```

Paraphrases are marked `reviewed=false` and `notes` explain origin. Prefer promoting paraphrases to reviewed only after a human look.
