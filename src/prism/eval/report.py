"""Write eval artifacts under reports/<run_id>/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"


def new_run_id(name: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = re_sub_name(name)
    return f"{ts}_{safe}"


def re_sub_name(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:40] or "run"


def write_report(run_id: str, summary: dict, per_question: list[dict]) -> Path:
    out = REPORTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for row in per_question:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return out


def _markdown(summary: dict) -> str:
    lines = [
        f"# Eval run `{summary.get('run_id')}`",
        "",
        f"- config: `{summary.get('eval_config')}`",
        f"- rag: `{summary.get('rag_config')}`",
        f"- dataset: `{summary.get('dataset')}`",
        f"- questions: {summary.get('n_questions')}",
        f"- generation: {summary.get('generation_enabled')}",
    ]
    prov = summary.get("provenance") or {}
    if prov.get("commit"):
        dirty = " dirty" if prov.get("dirty") else ""
        lines += [
            f"- git: `{prov.get('commit_short')}` ({prov.get('branch')}){dirty}",
            f"- describe: `{prov.get('describe')}`",
        ]
    lines += [
        "",
        "## Retrieval (answerable only)",
        "",
        "| metric | mean | 95% CI |",
        "|---|---:|---:|",
    ]
    ret = summary.get("retrieval", {})
    for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hit@5"):
        if key not in ret:
            continue
        m = ret[key]
        lines.append(f"| {key} | {m['mean']:.4f} | [{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}] |")
    lines += ["", "## Generation", ""]
    gen = summary.get("generation", {})
    if not gen:
        lines.append("_skipped_")
    else:
        lines += [
            "| metric | mean | 95% CI |",
            "|---|---:|---:|",
        ]
        for key in (
            "faithfulness",
            "answer_relevance",
            "citation_structural_ok",
            "correctness_lexical",
            "refusal_accuracy",
        ):
            if key not in gen or gen[key] is None:
                continue
            m = gen[key]
            lines.append(f"| {key} | {m['mean']:.4f} | [{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}] |")
    ops = summary.get("ops", {})
    lines += [
        "",
        "## Ops",
        "",
        f"- retrieve p50/p95 ms: {ops.get('retrieve_p50_ms')} / {ops.get('retrieve_p95_ms')}",
        f"- generate p50/p95 ms: {ops.get('generate_p50_ms')} / {ops.get('generate_p95_ms')}",
        f"- tokens in/out: {ops.get('tokens_in')} / {ops.get('tokens_out')}",
        f"- est. $/1k queries: {ops.get('usd_per_1k')}",
        "",
    ]
    return "\n".join(lines) + "\n"
