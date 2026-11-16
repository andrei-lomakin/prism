"""Post-generation grounding checks.

1) Every [n] must exist.
2) Numbers and Capitalized/Cyrillic names from the answer should appear in the
   cited chunk text (soft check — violations listed, ok=False if any hard miss).
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

    unsupported_numbers: list[str] = []
    unsupported_names: list[str] = []

    if sources and cited and not missing:
        # Strip citation markers before extracting entities
        clean = _CITE_RE.sub(" ", answer)
        for n_idx in cited:
            if n_idx < 1 or n_idx > len(sources):
                continue
            chunk = sources[n_idx - 1]
            chunk_l = chunk.lower()
            # numbers near this citation: take sentence containing [n]
            # simpler: all numbers in answer must appear in union of cited chunks
        cited_text = " ".join(
            sources[i - 1] for i in cited if 1 <= i <= len(sources)
        ).lower()
        for num in _NUM_RE.findall(clean):
            if num.lower() not in cited_text and num not in cited_text:
                unsupported_numbers.append(num)
        for name in list(_NAME_RE.findall(clean)) + list(_NAME_RU_RE.findall(clean)):
            # skip common non-entities
            if name.lower() in {"the", "this", "that", "rag", "bm25", "json"}:
                continue
            if name.lower() not in cited_text and name not in sources:
                # case-insensitive containment
                if name.lower() not in cited_text:
                    unsupported_names.append(name)

    ok = len(missing) == 0 and len(unsupported_numbers) == 0
    # names are soft: don't fail ok solely on names (too noisy), but report them
    return GroundingReport(
        ok=ok,
        cited=cited,
        missing=missing,
        unused_sources=unused,
        unsupported_numbers=sorted(set(unsupported_numbers)),
        unsupported_names=sorted(set(unsupported_names))[:20],
    )
