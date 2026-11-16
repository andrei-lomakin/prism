"""Post-generation citation check (S6-lite).

Stage 0: verify that every [n] exists in the provided source list.
Stage 3+: also check numbers/names appear in cited chunks + faithfulness judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GroundingReport:
    ok: bool
    cited: list[int]
    missing: list[int]
    unused_sources: list[int]


_CITE_RE = re.compile(r"\[(\d+)\]")


def check_citations(answer: str, n_sources: int) -> GroundingReport:
    cited = sorted({int(m.group(1)) for m in _CITE_RE.finditer(answer)})
    missing = [n for n in cited if n < 1 or n > n_sources]
    used = {n for n in cited if 1 <= n <= n_sources}
    unused = [i for i in range(1, n_sources + 1) if i not in used]
    return GroundingReport(
        ok=len(missing) == 0,
        cited=cited,
        missing=missing,
        unused_sources=unused,
    )
