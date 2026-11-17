"""Stylish printable PDF summary for an eval run (Cyrillic TTF)."""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Print-friendly palette (no purple glow nonsense)
INK = (15, 23, 36)
MUTED = (90, 106, 118)
LINE = (214, 222, 228)
CARD = (244, 247, 249)
ACCENT = (20, 104, 108)
ACCENT_SOFT = (232, 242, 243)
WARN = (180, 83, 9)
WHITE = (255, 255, 255)
BAR_TRACK = (226, 232, 236)


def _find_font_pair() -> tuple[Path, Path | None] | None:
    root = Path(__file__).resolve().parent / "fonts"
    bundled = root / "PrismSans.ttf"
    bundled_b = root / "PrismSans-Bold.ttf"
    regulars = [
        bundled,
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    bolds = [
        bundled_b,
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    regular = next((p for p in regulars if p.is_file() and p.stat().st_size > 1000), None)
    if regular is None:
        return None
    bold = next((p for p in bolds if p.is_file() and p.stat().st_size > 1000), None)
    return regular, bold


def _find_font() -> Path | None:
    pair = _find_font_pair()
    return pair[0] if pair else None


def _mean(block: Any) -> float | None:
    if block is None:
        return None
    if isinstance(block, dict) and "mean" in block:
        try:
            return float(block["mean"])
        except (TypeError, ValueError):
            return None
    if isinstance(block, (int, float)):
        return float(block)
    return None


def _fmt(v: Any, *, digits: int = 3) -> str:
    m = _mean(v) if not isinstance(v, (int, float)) else float(v)
    if m is None:
        return "—"
    return f"{m:.{digits}f}"


def _fmt_ci(block: Any) -> str:
    if not isinstance(block, dict):
        return "—"
    lo, hi = block.get("ci95_lo"), block.get("ci95_hi")
    if lo is None or hi is None:
        return "—"
    try:
        return f"[{float(lo):.3f}, {float(hi):.3f}]"
    except (TypeError, ValueError):
        return "—"


def _wrap_soft(s: Any, *, max_len: int = 0) -> str:
    text = str(s or "").replace("\\", "/")
    if max_len and len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _parse_run_stamp(run_id: str) -> str:
    head = (run_id or "").split("_", 1)[0]
    try:
        dt = datetime.strptime(head, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return "—"


def _role_label(summary: dict[str, Any]) -> str:
    gen = summary.get("generation_enabled")
    if gen:
        return "scientific · generation on"
    return "diagnostic · retrieval"


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

    pair = _find_font_pair()
    if pair is None:
        logger.warning("No TTF font with Cyrillic support found — skip PDF")
        return None
    font_reg, font_bold = pair

    rows = list(per_question or [])
    run_id = str(summary.get("run_id") or out_dir.name)
    rag = summary.get("rag") or {}
    ret = summary.get("retrieval") or {}
    ops = summary.get("ops") or {}
    gen = summary.get("generation") or {}
    prov = summary.get("provenance") or {}

    class Report(FPDF):
        def header(self) -> None:  # noqa: A003
            if self.page_no() == 1:
                return
            self.set_y(8)
            self.set_font("Prism", "B", 8)
            self.set_text_color(*MUTED)
            self.cell(self.epw * 0.55, 5, "PRISM · отчёт оценки", align="L")
            self.set_font("Prism", "", 8)
            self.cell(self.epw * 0.45, 5, _wrap_soft(run_id, max_len=42), align="R")
            self.ln(6)
            self.set_draw_color(*LINE)
            self.set_line_width(0.2)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

        def footer(self) -> None:
            self.set_y(-12)
            self.set_draw_color(*LINE)
            self.set_line_width(0.2)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(2)
            self.set_font("Prism", "", 8)
            self.set_text_color(*MUTED)
            git = prov.get("commit_short") or "—"
            dirty = " · dirty" if prov.get("dirty") else ""
            left = f"git {git}{dirty}"
            if prov.get("branch"):
                left += f" · {prov['branch']}"
            self.cell(self.epw * 0.7, 5, left, align="L")
            self.cell(self.epw * 0.3, 5, f"{self.page_no()}", align="R")

    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(14, 14, 14)
    pdf.add_font("Prism", "", str(font_reg))
    if font_bold is not None:
        pdf.add_font("Prism", "B", str(font_bold))
    else:
        pdf.add_font("Prism", "B", str(font_reg))

    # ── Cover band ──────────────────────────────────────────────
    pdf.add_page()
    band_h = 42
    y0 = pdf.get_y() - 4
    pdf.set_fill_color(*INK)
    pdf.rect(0, 0, pdf.w, band_h + 6, style="F")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(0, band_h + 6, pdf.w, 2.2, style="F")

    pdf.set_xy(14, 10)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Prism", "B", 22)
    pdf.cell(60, 10, "PRISM")
    pdf.set_font("Prism", "", 10)
    pdf.set_xy(14, 22)
    pdf.set_text_color(180, 198, 204)
    pdf.cell(120, 6, "отчёт оценки retrieval / RAG")
    pdf.set_xy(14, 30)
    pdf.set_font("Prism", "B", 11)
    pdf.set_text_color(*WHITE)
    pdf.cell(pdf.epw, 6, _wrap_soft(run_id, max_len=70))

    pdf.set_xy(pdf.w - 70, 12)
    pdf.set_font("Prism", "", 9)
    pdf.set_text_color(180, 198, 204)
    pdf.cell(56, 5, _parse_run_stamp(run_id), align="R")
    pdf.set_xy(pdf.w - 70, 18)
    pdf.set_font("Prism", "B", 9)
    pdf.set_text_color(*WHITE)
    pdf.cell(56, 5, _role_label(summary), align="R")
    pdf.set_xy(pdf.w - 70, 24)
    pdf.set_font("Prism", "", 9)
    pdf.set_text_color(180, 198, 204)
    n_q = summary.get("n_questions")
    pdf.cell(56, 5, f"{n_q if n_q is not None else '—'} вопросов", align="R")

    pdf.set_y(band_h + 14)

    def section(title: str) -> None:
        pdf.ln(2)
        if pdf.get_y() > pdf.h - 40:
            pdf.add_page()
        pdf.set_font("Prism", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(pdf.epw, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(*ACCENT)
        pdf.set_line_width(0.6)
        y = pdf.get_y()
        pdf.line(pdf.l_margin, y, pdf.l_margin + 28, y)
        pdf.set_line_width(0.2)
        pdf.ln(4)

    def kv_row(label: str, value: str, *, zebra: bool = False) -> None:
        h = 6.2
        x = pdf.l_margin
        w_l, w_v = 38, pdf.epw - 38
        if zebra:
            pdf.set_fill_color(*CARD)
            pdf.rect(x, pdf.get_y(), pdf.epw, h, style="F")
        pdf.set_xy(x, pdf.get_y())
        pdf.set_font("Prism", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(w_l, h, label)
        pdf.set_font("Prism", "", 9)
        pdf.set_text_color(*INK)
        pdf.cell(w_v, h, _wrap_soft(value, max_len=78), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── KPI cards ───────────────────────────────────────────────
    section("Ключевые метрики")
    kpis = [
        ("Recall@5", _fmt(ret.get("recall@5")), _fmt_ci(ret.get("recall@5"))),
        ("MRR@10", _fmt(ret.get("mrr@10")), _fmt_ci(ret.get("mrr@10"))),
        ("nDCG@10", _fmt(ret.get("ndcg@10")), _fmt_ci(ret.get("ndcg@10"))),
        ("Hit@5", _fmt(ret.get("hit@5")), _fmt_ci(ret.get("hit@5"))),
        (
            "p95 retrieve",
            f"{ops.get('retrieve_p95_ms') if ops.get('retrieve_p95_ms') is not None else '—'} мс",
            "latency",
        ),
        ("$/1k", f"{ops.get('usd_per_1k') if ops.get('usd_per_1k') is not None else '—'}", "cost"),
    ]
    gap = 3.0
    card_w = (pdf.epw - gap * 2) / 3
    card_h = 22.0
    start_y = pdf.get_y()
    for i, (label, value, sub) in enumerate(kpis):
        col = i % 3
        row = i // 3
        x = pdf.l_margin + col * (card_w + gap)
        y = start_y + row * (card_h + gap)
        pdf.set_fill_color(*CARD)
        pdf.set_draw_color(*LINE)
        pdf.rect(x, y, card_w, card_h, style="DF")
        pdf.set_fill_color(*ACCENT)
        pdf.rect(x, y, 1.6, card_h, style="F")
        pdf.set_xy(x + 5, y + 3.5)
        pdf.set_font("Prism", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(card_w - 8, 4, label)
        pdf.set_xy(x + 5, y + 8.5)
        pdf.set_font("Prism", "B", 16)
        pdf.set_text_color(*INK)
        pdf.cell(card_w - 8, 7, str(value))
        pdf.set_xy(x + 5, y + 16)
        pdf.set_font("Prism", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(
            card_w - 8,
            4,
            sub
            if sub not in ("latency", "cost")
            else ("95-й перцентиль" if sub == "latency" else "оценка стоимости"),
        )
    pdf.set_y(start_y + 2 * (card_h + gap) + 2)

    # ── Config ──────────────────────────────────────────────────
    section("Конфигурация")
    kv_row("Eval config", _wrap_soft(summary.get("eval_config")), zebra=True)
    kv_row("RAG config", _wrap_soft(summary.get("rag_config")))
    kv_row("Dataset", _wrap_soft(summary.get("dataset")), zebra=True)
    kv_row("Retrieval", str(rag.get("retrieval") or "—"))
    kv_row("Chunking", str(rag.get("chunking") or "—"), zebra=True)
    kv_row("Embedding", _wrap_soft(rag.get("embedding") or "—", max_len=70))
    kv_row("Rewrite / rerank", f"{rag.get('rewrite') or '—'} / {rag.get('rerank')}", zebra=True)
    kv_row("top_k eval", str(rag.get("top_k_eval") if rag.get("top_k_eval") is not None else "—"))
    kv_row("Generation", "on" if summary.get("generation_enabled") else "off", zebra=True)

    # ── Retrieval table ─────────────────────────────────────────
    section("Retrieval · mean и 95% CI")
    metric_keys = [
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "precision@5",
        "context_precision@5",
        "context_recall@5",
        "mrr@10",
        "ndcg@10",
        "hit@5",
    ]
    # header
    col_m, col_v, col_ci, col_bar = 48, 22, 40, pdf.epw - 48 - 22 - 40
    pdf.set_fill_color(*INK)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Prism", "B", 8)
    y = pdf.get_y()
    pdf.rect(pdf.l_margin, y, pdf.epw, 7, style="F")
    pdf.set_xy(pdf.l_margin + 2, y + 1)
    pdf.cell(col_m - 2, 5, "метрика")
    pdf.cell(col_v, 5, "mean", align="R")
    pdf.cell(col_ci, 5, "95% CI", align="C")
    pdf.cell(col_bar, 5, "шкала 0–1", align="C")
    pdf.set_y(y + 7)

    def bar(x: float, y: float, w: float, h: float, frac: float | None) -> None:
        pdf.set_fill_color(*BAR_TRACK)
        pdf.rect(x, y, w, h, style="F")
        if frac is None:
            return
        frac = max(0.0, min(1.0, frac))
        color = ACCENT if frac >= 0.45 else WARN
        pdf.set_fill_color(*color)
        if frac > 0:
            pdf.rect(x, y, max(0.8, w * frac), h, style="F")

    zebra = False
    for key in metric_keys:
        if key not in ret:
            continue
        block = ret[key]
        mean = _mean(block)
        h = 7.2
        if pdf.get_y() + h > pdf.h - 20:
            pdf.add_page()
        y = pdf.get_y()
        if zebra:
            pdf.set_fill_color(*CARD)
            pdf.rect(pdf.l_margin, y, pdf.epw, h, style="F")
        zebra = not zebra
        pdf.set_xy(pdf.l_margin + 2, y + 1.2)
        pdf.set_font("Prism", "", 9)
        pdf.set_text_color(*INK)
        pdf.cell(col_m - 2, 5, key)
        pdf.set_font("Prism", "B", 9)
        pdf.cell(col_v, 5, _fmt(block), align="R")
        pdf.set_font("Prism", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(col_ci, 5, _fmt_ci(block), align="C")
        bar(pdf.l_margin + col_m + col_v + col_ci + 2, y + 2.2, col_bar - 6, 3.2, mean)
        pdf.set_y(y + h)

    # ── Slice by type / language ────────────────────────────────
    answerable = [r for r in rows if r.get("type") != "unanswerable"]
    if answerable:
        section("Срезы датасета")
        by_type: dict[str, list[float]] = defaultdict(list)
        by_lang: dict[str, list[float]] = defaultdict(list)
        latencies: list[float] = []
        for r in answerable:
            v = r.get("recall@5")
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            by_type[str(r.get("type") or "?")].append(fv)
            by_lang[str(r.get("language") or "?")].append(fv)
            if r.get("retrieve_ms") is not None:
                try:
                    latencies.append(float(r["retrieve_ms"]))
                except (TypeError, ValueError):
                    pass

        type_counts = Counter(str(r.get("type") or "?") for r in rows)
        pdf.set_font("Prism", "B", 9)
        pdf.set_text_color(*INK)
        pdf.cell(
            pdf.epw, 5, "По типу вопроса (Recall@5 mean · n)", new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        pdf.ln(1)

        types_sorted = sorted(by_type.keys())
        tw = (pdf.epw - 2 * 2) / max(1, min(4, len(types_sorted) or 1))
        # also show unanswerable count as pill
        chips_y = pdf.get_y()
        for i, t in enumerate(types_sorted[:8]):
            vals = by_type[t]
            mean = statistics.fmean(vals) if vals else 0.0
            n = type_counts.get(t, len(vals))
            col = i % 4
            row = i // 4
            x = pdf.l_margin + col * (tw + 2)
            y = chips_y + row * 16
            pdf.set_fill_color(*ACCENT_SOFT)
            pdf.set_draw_color(*LINE)
            pdf.rect(x, y, tw, 14, style="DF")
            pdf.set_xy(x + 3, y + 2)
            pdf.set_font("Prism", "", 8)
            pdf.set_text_color(*MUTED)
            pdf.cell(tw - 6, 4, f"{t} · n={n}")
            pdf.set_xy(x + 3, y + 6.5)
            pdf.set_font("Prism", "B", 11)
            pdf.set_text_color(*INK)
            pdf.cell(tw - 6, 5, f"{mean:.3f}")
        rows_chip = (min(8, len(types_sorted)) + 3) // 4
        pdf.set_y(chips_y + max(1, rows_chip) * 16 + 3)

        # language + latency strip
        pdf.set_fill_color(*CARD)
        strip_h = 16
        y = pdf.get_y()
        pdf.rect(pdf.l_margin, y, pdf.epw, strip_h, style="F")
        pdf.set_xy(pdf.l_margin + 4, y + 3)
        pdf.set_font("Prism", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(30, 4, "Языки")
        pdf.set_xy(pdf.l_margin + 4, y + 8)
        pdf.set_font("Prism", "B", 10)
        pdf.set_text_color(*INK)
        lang_bits = []
        for lang, vals in sorted(by_lang.items()):
            lang_bits.append(f"{lang}: {statistics.fmean(vals):.3f} (n={len(vals)})")
        pdf.cell(pdf.epw * 0.55, 5, "  ·  ".join(lang_bits) or "—")

        pdf.set_xy(pdf.l_margin + pdf.epw * 0.58, y + 3)
        pdf.set_font("Prism", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(40, 4, "retrieve latency")
        pdf.set_xy(pdf.l_margin + pdf.epw * 0.58, y + 8)
        pdf.set_font("Prism", "B", 10)
        pdf.set_text_color(*INK)
        if latencies:
            p50 = statistics.median(latencies)
            lat_sorted = sorted(latencies)
            p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * (len(lat_sorted) - 1)))]
            pdf.cell(60, 5, f"p50 {p50:.0f} · p95 {p95:.0f} мс")
        else:
            pdf.cell(60, 5, "—")
        pdf.set_y(y + strip_h + 2)

        ua = type_counts.get("unanswerable", 0)
        if ua:
            pdf.set_font("Prism", "", 8)
            pdf.set_text_color(*MUTED)
            pdf.cell(
                pdf.epw,
                5,
                f"unanswerable в датасете: {ua} (не входят в retrieval mean)",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

    # ── Ops ─────────────────────────────────────────────────────
    section("Ops")
    kv_row(
        "retrieve p50 / p95",
        f"{ops.get('retrieve_p50_ms')} / {ops.get('retrieve_p95_ms')} мс",
        zebra=True,
    )
    kv_row("generate p50 / p95", f"{ops.get('generate_p50_ms')} / {ops.get('generate_p95_ms')} мс")
    kv_row("tokens in / out", f"{ops.get('tokens_in')} / {ops.get('tokens_out')}", zebra=True)
    kv_row("$ / 1k queries", str(ops.get("usd_per_1k")))

    if gen:
        section("Generation")
        for key in ("faithfulness", "answer_relevance", "citation_structural_ok"):
            if key in gen and gen[key] is not None:
                kv_row(key, f"{_fmt(gen[key])}  {_fmt_ci(gen[key])}", zebra=True)

    # ── Worst questions ─────────────────────────────────────────
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in answerable:
        val = r.get("recall@5")
        if val is None:
            continue
        try:
            scored.append((float(val), r))
        except (TypeError, ValueError):
            continue
    scored.sort(key=lambda x: x[0])
    worst = scored[:8]
    best = list(reversed(scored[-3:])) if len(scored) >= 3 else []

    if worst:
        section("Худшие вопросы · Recall@5")
        # table header
        cols = [22, 22, 16, 18, pdf.epw - 22 - 22 - 16 - 18]
        headers = ["id", "type", "R@5", "ms", "вопрос"]
        y = pdf.get_y()
        pdf.set_fill_color(*INK)
        pdf.rect(pdf.l_margin, y, pdf.epw, 6.5, style="F")
        pdf.set_text_color(*WHITE)
        pdf.set_font("Prism", "B", 7.5)
        pdf.set_xy(pdf.l_margin + 1.5, y + 1)
        for w, h in zip(cols, headers):
            pdf.cell(w, 4.5, h)
        pdf.set_y(y + 6.5)

        zebra = False
        for val, r in worst:
            q = _wrap_soft(r.get("question") or "", max_len=78)
            rid = _wrap_soft(r.get("id") or "", max_len=18)
            typ = _wrap_soft(r.get("type") or "", max_len=14)
            ms = r.get("retrieve_ms")
            ms_s = f"{float(ms):.0f}" if ms is not None else "—"
            # estimate height
            pdf.set_font("Prism", "", 8)
            # force single-line for compactness
            h = 7.0
            if pdf.get_y() + h > pdf.h - 20:
                pdf.add_page()
            y = pdf.get_y()
            if zebra:
                pdf.set_fill_color(*CARD)
                pdf.rect(pdf.l_margin, y, pdf.epw, h, style="F")
            zebra = not zebra
            pdf.set_xy(pdf.l_margin + 1.5, y + 1.5)
            pdf.set_font("Prism", "", 7.5)
            pdf.set_text_color(*MUTED)
            pdf.cell(cols[0], 4, rid)
            pdf.set_text_color(*INK)
            pdf.cell(cols[1], 4, typ)
            pdf.set_font("Prism", "B", 8)
            color = WARN if val < 0.35 else INK
            pdf.set_text_color(*color)
            pdf.cell(cols[2], 4, f"{val:.3f}")
            pdf.set_font("Prism", "", 7.5)
            pdf.set_text_color(*MUTED)
            pdf.cell(cols[3], 4, ms_s)
            pdf.set_text_color(*INK)
            pdf.cell(cols[4], 4, q)
            pdf.set_y(y + h)

    if best:
        section("Лучшие вопросы · Recall@5")
        for val, r in best:
            if pdf.get_y() > pdf.h - 24:
                pdf.add_page()
            pdf.set_fill_color(*ACCENT_SOFT)
            y = pdf.get_y()
            pdf.rect(pdf.l_margin, y, pdf.epw, 11, style="F")
            pdf.set_xy(pdf.l_margin + 3, y + 1.5)
            pdf.set_font("Prism", "B", 9)
            pdf.set_text_color(*ACCENT)
            pdf.cell(18, 4, f"{val:.3f}")
            pdf.set_text_color(*MUTED)
            pdf.set_font("Prism", "", 8)
            pdf.cell(40, 4, f"{r.get('id')} · {r.get('type')}")
            pdf.set_xy(pdf.l_margin + 3, y + 6)
            pdf.set_text_color(*INK)
            pdf.cell(pdf.epw - 6, 4, _wrap_soft(r.get("question") or "", max_len=95))
            pdf.set_y(y + 13)

    # closing note
    pdf.ln(4)
    pdf.set_font("Prism", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        pdf.epw,
        4,
        "Метрики retrieval считаются по answerable вопросам. "
        "Полосы — mean на шкале 0–1. CI — bootstrap/нормальный 95% из runner.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    out = out_dir / "report.pdf"
    pdf.output(str(out))
    return out
