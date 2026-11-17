"""Post-generation grounding checks.

Structural / lexical heuristics — NOT full semantic entailment.

Hard fail when:
1) a citation [n] is out of range;
2) sources were provided but the answer has no valid citation at all;
3) a number in the answer is missing from the union of *cited* chunk texts.

Names are soft diagnostics only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GroundingReport:
    ok: bool
    cited: list[int]
    missing: list[int]
    unused_sources: list[int]
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_names: list[str] = field(default_factory=list)
    requires_citation: bool = False


_CITE_RE = re.compile(r"\[(\d+)\]")
_NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?")
_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\b")
_NAME_RU_RE = re.compile(r"\b([А-ЯЁ][а-яёА-ЯЁ]{2,}(?:\s+[А-ЯЁ][а-яёА-ЯЁ]{2,})?)\b")


def check_citations(answer: str, n_sources: int) -> GroundingReport:
    return check_grounding(answer, [], n_sources)


def check_grounding(
    answer: str,
    sources: list[str],
    n_sources: int | None = None,
) -> GroundingReport:
    n = n_sources if n_sources is not None else len(sources)
    cited = sorted({int(m.group(1)) for m in _CITE_RE.finditer(answer)})
    missing = [i for i in cited if i < 1 or i > n]
    used = {i for i in cited if 1 <= i <= n}
    unused = [i for i in range(1, n + 1) if i not in used]

    # If the retriever gave context, an answer with zero valid citations is not grounded.
    requires_citation = n > 0
    has_valid_citation = len(used) > 0

    unsupported_numbers: list[str] = []
    unsupported_names: list[str] = []

    if sources and cited and not missing:
        clean = _CITE_RE.sub(" ", answer)
        cited_text = " ".join(sources[i - 1] for i in cited if 1 <= i <= len(sources)).lower()
        for num in _NUM_RE.findall(clean):
            if num.lower() not in cited_text and num not in cited_text:
                unsupported_numbers.append(num)
        for name in list(_NAME_RE.findall(clean)) + list(_NAME_RU_RE.findall(clean)):
            if name.lower() in {"the", "this", "that", "rag", "bm25", "json"}:
                continue
            if name.lower() not in cited_text:
                unsupported_names.append(name)

    ok = (
        len(missing) == 0
        and len(unsupported_numbers) == 0
        and (has_valid_citation if requires_citation else True)
    )
    return GroundingReport(
        ok=ok,
        cited=cited,
        missing=missing,
        unused_sources=unused,
        unsupported_numbers=sorted(set(unsupported_numbers)),
        unsupported_names=sorted(set(unsupported_names))[:20],
        requires_citation=requires_citation,
    )
