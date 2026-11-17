"""Generate a printable PDF summary for an eval run (Cyrillic-capable fonts)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _find_font() -> Path | None:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _mean(block: Any) -> str:
    if block is None:
        return "—"
    if isinstance(block, dict) and "mean" in block:
        return f"{float(block['mean']):.4f}"
    if isinstance(block, (int, float)):
        return f"{float(block):.4f}"
    return "—"


def _wrap_path(s: Any) -> str:
    """Allow fpdf to break long paths / urls."""
    text = str(s or "").replace("\\", "/")
    return text.replace("/", "/ ").replace("_", "_ ")


def write_pdf_report(
    out_dir: Path,
    summary: dict[str, Any],
    per_question: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Write ``report.pdf`` next to metrics.json. Returns path or None if fpdf missing."""
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError:
        logger.warning("fpdf2 not installed — skip PDF report")
        return None

    font = _find_font()
    if font is None:
        logger.warning("No TTF font with Cyrillic support found — skip PDF")
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.add_font("Prism", "", str(font))
    pdf.set_font("Prism", size=16)
    run_id = summary.get("run_id") or out_dir.name
    pdf.multi_cell(
        pdf.epw, 9, f"Prism · отчёт оценки\n{_wrap_path(run_id)}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(4)

    def line(text: str, *, size: int = 11, h: float = 6) -> None:
        pdf.set_font("Prism", size=size)
        pdf.multi_cell(pdf.epw, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    line(f"Конфиг eval: {_wrap_path(summary.get('eval_config'))}")
    line(f"Конфиг RAG: {_wrap_path(summary.get('rag_config'))}")
    line(f"Датасет: {_wrap_path(summary.get('dataset'))}")
    line(f"Вопросов: {summary.get('n_questions')}")
    line(f"Генерация: {summary.get('generation_enabled')}")
    rag = summary.get("rag") or {}
    line(
        "Стек: "
        f"retrieval={rag.get('retrieval')} · chunking={rag.get('chunking')} · "
        f"rewrite={rag.get('rewrite')} · rerank={rag.get('rerank')}"
    )
    prov = summary.get("provenance") or {}
    if prov.get("commit_short"):
        dirty = " (dirty)" if prov.get("dirty") else ""
        line(f"Git: {prov.get('commit_short')}{dirty} · {prov.get('branch')}")
    pdf.ln(3)

    line("Retrieval", size=13, h=8)
    ret = summary.get("retrieval") or {}
    for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hit@5"):
        if key in ret:
            m = ret[key] if isinstance(ret[key], dict) else {"mean": ret[key]}
            ci = ""
            if isinstance(m, dict) and "ci95_lo" in m:
                ci = f"  [{m['ci95_lo']:.3f}, {m['ci95_hi']:.3f}]"
            line(f"{key}: {_mean(m)}{ci}")

    ops = summary.get("ops") or {}
    pdf.ln(2)
    line("Ops", size=13, h=8)
    line(f"retrieve p50/p95: {ops.get('retrieve_p50_ms')} / {ops.get('retrieve_p95_ms')} мс")
    line(f"$/1k: {ops.get('usd_per_1k')}")

    gen = summary.get("generation") or {}
    if gen:
        pdf.ln(2)
        line("Generation", size=13, h=8)
        for key in ("faithfulness", "answer_relevance", "citation_structural_ok"):
            if key in gen and gen[key] is not None:
                line(f"{key}: {_mean(gen[key])}")

    rows = list(per_question or [])
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        if r.get("type") == "unanswerable":
            continue
        val = r.get("recall@5")
        if val is None:
            continue
        scored.append((float(val), r))
    scored.sort(key=lambda x: x[0])
    worst = scored[:5]
    if worst:
        pdf.ln(2)
        line("Худшие вопросы (Recall@5)", size=13, h=8)
        for val, r in worst:
            q = (r.get("question") or "")[:160]
            line(f"• {r.get('id')} · {val:.3f}", size=10, h=5)
            line(f"  {q}", size=10, h=5)

    out = out_dir / "report.pdf"
    pdf.output(str(out))
    return out
