"""Load golden JSONL for eval runs."""

from __future__ import annotations

import json
from pathlib import Path

from prism.eval.schema import GoldenQuestion

ROOT = Path(__file__).resolve().parents[3]


def resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_golden(path: str | Path, *, max_questions: int | None = None) -> list[GoldenQuestion]:
    file_path = resolve_data_path(path)
    items: list[GoldenQuestion] = []
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = GoldenQuestion.model_validate(json.loads(line))
            q.validate_consistency()
            items.append(q)
            if max_questions is not None and len(items) >= max_questions:
                break
    return items
