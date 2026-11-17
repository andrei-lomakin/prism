"""Eval runner: golden → retrieve → (optional generate+judge) → reports.

Baseline = C2 + multilingual MiniLM + dense (configs/rag/baseline.yaml).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from prism.config import Settings, get_settings
from prism.db.session import get_session_factory, init_db
from prism.eval.aggregate import summarize_metric
from prism.eval.dataset import load_golden, resolve_data_path
from prism.eval.judge import judge_answer
from prism.eval.metrics import (
    answer_correctness_lexical,
    chunk_relevance_labels,
    context_precision_at_k,
    context_recall_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from prism.eval.provenance import git_provenance
from prism.eval.relevance import (
    gold_doc_keys,
    gold_evidence_units,
    retrieved_context_units,
    retrieved_doc_keys,
)
from prism.eval.report import new_run_id, write_report
from prism.eval.schema import GoldenQuestion
from prism.llm.gateway import LLMGateway
from prism.rag.generate import SYSTEM_PROMPT, build_rag_prompt
from prism.rag.retrieve import retrieve
from prism.rag.retrieve.packer import pack_context

ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: str | Path) -> dict:
    p = resolve_data_path(path) if not Path(path).is_absolute() else Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / path
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _settings_from_rag_yaml(rag: dict, base: Settings | None = None) -> Settings:
    base = base or get_settings()
    emb = (rag.get("embedding") or {}).get("model") or base.embedding_model
    ch = rag.get("chunking") or {}
    ret = rag.get("retrieval") or {}
    pack = rag.get("packer") or {}
    gen = rag.get("generation") or {}
    rerank = rag.get("rerank") or {}
    rewrite = rag.get("rewrite") or {}
    return base.model_copy(
        update={
            "embedding_model": emb,
            "chunking_strategy": ch.get("strategy", base.chunking_strategy),
            "chunk_size": int(ch.get("size", base.chunk_size)),
            "chunk_overlap": int(ch.get("overlap", base.chunk_overlap)),
            "retrieval_mode": ret.get("mode", base.retrieval_mode),
            "top_k": int(ret.get("top_k", base.top_k)),
            "hybrid_fusion": ret.get("fusion", base.hybrid_fusion),
            "hybrid_alpha": float(ret.get("alpha", base.hybrid_alpha)),
            "rrf_k": int(ret.get("rrf_k", base.rrf_k)),
            "rewrite_strategy": rewrite.get("strategy", base.rewrite_strategy),
            "rerank_enabled": bool(rerank.get("enabled", base.rerank_enabled)),
            "rerank_model": rerank.get("model", base.rerank_model),
            "rerank_candidates": int(rerank.get("candidates", base.rerank_candidates)),
            "pack_max_chunks": int(pack.get("max_chunks", base.pack_max_chunks)),
            "pack_max_tokens": int(pack.get("max_tokens", base.pack_max_tokens)),
            "pack_max_per_doc": int(pack.get("max_per_doc", base.pack_max_per_doc)),
            "pack_min_score": float(pack.get("min_score", base.pack_min_score)),
            "llm_provider": gen.get("provider", base.llm_provider),
            "llm_model": gen.get("model", base.llm_model),
        }
    )


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((p / 100) * (len(ys) - 1)))))
    return float(ys[idx])


def run_eval(
    eval_config_path: str,
    *,
    skip_generation: bool = False,
    use_llm_judge: bool = True,
) -> dict[str, Any]:
    cfg = _load_yaml(eval_config_path)
    rag_path = cfg.get("config") or "configs/rag/baseline.yaml"
    rag = _load_yaml(rag_path)
    settings = _settings_from_rag_yaml(rag)

    dataset_path = cfg.get("dataset") or "data/golden/dev.jsonl"
    max_q = cfg.get("max_questions")
    questions = load_golden(dataset_path, max_questions=max_q)

    gen_enabled = bool(cfg.get("generation", True)) and not skip_generation
    if cfg.get("retrieval_only"):
        gen_enabled = False

    init_db()
    db = get_session_factory()()

    run_id = new_run_id(cfg.get("name") or "eval")
    per_q: list[dict] = []

    # metric accumulators
    r1, r3, r5, r10, mrr10, ndcg10, hit5 = [], [], [], [], [], [], []
    p5, ctx_p5, ctx_r5 = [], [], []
    faith, arel, cite_acc, refuse, correct = [], [], [], [], []
    ret_ms, gen_ms = [], []
    tokens_in = tokens_out = 0

    # Eval retrieve width: need @10
    eval_top_k = max(10, int(settings.top_k))

    try:
        for q in questions:
            row = _eval_one(
                db,
                q,
                settings=settings,
                eval_top_k=eval_top_k,
                generation=gen_enabled,
                use_llm_judge=use_llm_judge,
            )
            per_q.append(row)
            ret_ms.append(row["retrieve_ms"])

            if q.type != "unanswerable":
                r1.append(row["recall@1"])
                r3.append(row["recall@3"])
                r5.append(row["recall@5"])
                r10.append(row["recall@10"])
                mrr10.append(row["mrr@10"])
                ndcg10.append(row["ndcg@10"])
                hit5.append(row["hit@5"])
                p5.append(row["precision@5"])
                ctx_p5.append(row["context_precision@5"])
                ctx_r5.append(row["context_recall@5"])

            if gen_enabled and row.get("generation"):
                g = row["generation"]
                faith.append(g["faithfulness"])
                arel.append(g["answer_relevance"])
                cite_acc.append(1.0 if g["citation_ok"] else 0.0)
                if g.get("correctness") is not None:
                    correct.append(float(g["correctness"]))
                if g.get("refusal_correct") is not None:
                    refuse.append(float(g["refusal_correct"]))
                gen_ms.append(g.get("latency_ms") or 0)
                tokens_in += int(g.get("tokens_in") or 0)
                tokens_out += int(g.get("tokens_out") or 0)
    finally:
        db.close()

    n_q = len(questions)
    # Local Ollama cost = 0; keep field for cloud runs later
    usd_per_1k = 0.0
    if n_q:
        # placeholder: if somehow cost tracked later
        usd_per_1k = round(1000.0 * 0.0 / n_q, 4)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "eval_config": str(eval_config_path),
        "rag_config": str(rag_path),
        "dataset": str(dataset_path),
        "n_questions": n_q,
        "generation_enabled": gen_enabled,
        "provenance": git_provenance(),
        "rag": {
            "chunking": settings.chunking_strategy,
            "embedding": settings.embedding_model,
            "retrieval": settings.retrieval_mode,
            "rewrite": settings.rewrite_strategy,
            "rerank": settings.rerank_enabled,
            "top_k_eval": eval_top_k,
        },
        "retrieval": {
            "recall@1": summarize_metric(r1),
            "recall@3": summarize_metric(r3),
            "recall@5": summarize_metric(r5),
            "recall@10": summarize_metric(r10),
            "precision@5": summarize_metric(p5),
            "context_precision@5": summarize_metric(ctx_p5),
            "context_recall@5": summarize_metric(ctx_r5),
            "mrr@10": summarize_metric(mrr10),
            "ndcg@10": summarize_metric(ndcg10),
            "hit@5": summarize_metric(hit5),
        },
        "ops": {
            "retrieve_p50_ms": round(_percentile(ret_ms, 50), 1),
            "retrieve_p95_ms": round(_percentile(ret_ms, 95), 1),
            "generate_p50_ms": round(_percentile(gen_ms, 50), 1) if gen_ms else None,
            "generate_p95_ms": round(_percentile(gen_ms, 95), 1) if gen_ms else None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "usd_per_1k": usd_per_1k,
        },
    }
    if gen_enabled:
        summary["generation"] = {
            "faithfulness": summarize_metric(faith),
            "answer_relevance": summarize_metric(arel),
            # Structural [n] check — not semantic citation accuracy.
            "citation_structural_ok": summarize_metric(cite_acc),
            "correctness_lexical": summarize_metric(correct) if correct else None,
            "refusal_accuracy": summarize_metric(refuse) if refuse else None,
            "judge_note": (
                "LLM-as-judge via Ollama when available; heuristic fallback on failure. "
                "citation_structural_ok = valid [n] targets (not semantic accuracy). "
                "correctness_lexical = token-F1 vs gold (not semantic correctness). "
                "Cohen's κ vs human labels: pending (see data/golden/ANNOTATION.md). "
                "Use `prism eval kappa --a ... --b ...` when dual annotations exist."
            ),
        }

    out_dir = write_report(run_id, summary, per_q)
    summary["report_dir"] = str(out_dir)
    return summary


def _eval_one(
    db,
    q: GoldenQuestion,
    *,
    settings: Settings,
    eval_top_k: int,
    generation: bool,
    use_llm_judge: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    # Score@k on raw ranked hits — packer must not shrink Recall@10.
    ranked_hits, mode = retrieve(db, q.question, top_k=eval_top_k, settings=settings, pack=False)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    relevant = gold_doc_keys(q)
    ranked = retrieved_doc_keys(ranked_hits)
    gold_units = gold_evidence_units(q)
    # Context_* = what generation would see after pack (section-aware).
    packed_hits = pack_context(
        ranked_hits,
        max_chunks=settings.pack_max_chunks,
        max_tokens=settings.pack_max_tokens,
        max_per_doc=settings.pack_max_per_doc,
        min_score=settings.pack_min_score,
    )
    ctx_units = retrieved_context_units(packed_hits)
    ctx_labels = chunk_relevance_labels(gold_units, ctx_units)

    row: dict[str, Any] = {
        "id": q.id,
        "type": q.type,
        "language": q.language,
        "question": q.question,
        "retrieval_mode": mode,
        "retrieve_ms": round(retrieve_ms, 2),
        "relevant_docs": sorted(relevant),
        "retrieved_docs": ranked,
        "n_ranked": len(ranked_hits),
        "n_packed": len(packed_hits),
        "recall@1": recall_at_k(relevant, ranked, 1) if relevant else None,
        "recall@3": recall_at_k(relevant, ranked, 3) if relevant else None,
        "recall@5": recall_at_k(relevant, ranked, 5) if relevant else None,
        "recall@10": recall_at_k(relevant, ranked, 10) if relevant else None,
        "precision@5": precision_at_k(relevant, ranked, 5) if relevant else None,
        "context_precision@5": context_precision_at_k(ctx_labels, 5) if gold_units else None,
        "context_recall@5": context_recall_at_k(gold_units, ctx_units, 5) if gold_units else None,
        "mrr@10": mrr_at_k(relevant, ranked, 10) if relevant else None,
        "ndcg@10": ndcg_at_k(relevant, ranked, 10) if relevant else None,
        "hit@5": hit_rate_at_k(relevant, ranked, 5) if relevant else None,
    }

    if not generation:
        return row

    contexts = [
        {
            "text": h.text,
            "document_name": h.document_name,
            "score": h.score,
            "section_path": h.section_path,
        }
        for h in packed_hits[: settings.top_k]
    ]
    prompt = build_rag_prompt(q.question, contexts)
    gateway = LLMGateway(settings)
    try:
        result = gateway.chat(user_message=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1)
        answer = result.text
        gen_latency = result.latency_ms
        tin, tout = result.tokens_in, result.tokens_out
    except Exception as exc:
        answer = f"[generation_failed] {exc}"
        gen_latency = 0
        tin = tout = 0

    scores = judge_answer(
        question=q.question,
        answer=answer,
        contexts=[c["text"] for c in contexts],
        qtype=q.type,
        settings=settings,
        use_llm=use_llm_judge,
    )
    correctness = None
    if q.type != "unanswerable" and q.gold_answer:
        correctness = answer_correctness_lexical(q.gold_answer, answer)
    row["generation"] = {
        "answer": answer,
        "faithfulness": scores.faithfulness,
        "answer_relevance": scores.answer_relevance,
        "citation_ok": scores.citation_ok,
        "refusal_correct": scores.refusal_correct,
        "correctness": correctness,
        "judge_method": scores.method,
        "latency_ms": gen_latency,
        "tokens_in": tin,
        "tokens_out": tout,
        "model": settings.llm_model,
    }
    return row
