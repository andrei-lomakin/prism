"""Inter-annotator / judge-agreement helpers (honest κ when two label files exist)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prism.eval.stats import cohens_kappa


def load_label_map(
    path: Path, *, id_field: str = "id", label_field: str = "label"
) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row[id_field])] = str(row[label_field])
    return out


def kappa_from_files(path_a: Path | str, path_b: Path | str) -> dict[str, Any]:
    a = load_label_map(Path(path_a))
    b = load_label_map(Path(path_b))
    ids = sorted(set(a) & set(b))
    la = [a[i] for i in ids]
    lb = [b[i] for i in ids]
    k = cohens_kappa(la, lb)
    return {"n_overlap": len(ids), "kappa": None if k is None else round(k, 4), "ids": ids[:20]}


def judge_method_agreement(per_question_path: Path | str) -> dict[str, Any]:
    """If a report mixes llm/heuristic judges, κ on binned faithfulness (high/low)."""
    path = Path(per_question_path)
    labels_m: list[str] = []
    labels_f: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            g = row.get("generation") or {}
            if not g:
                continue
            method = str(g.get("judge_method") or "unknown")
            faith = float(g.get("faithfulness") or 0.0)
            labels_m.append(method)
            labels_f.append("high" if faith >= 0.5 else "low")
    # Not true κ between annotators — report distribution only
    return {
        "n": len(labels_m),
        "methods": {m: labels_m.count(m) for m in set(labels_m)},
        "note": "Human κ still pending; use kappa_from_files(annotator1.jsonl, annotator2.jsonl)",
    }
