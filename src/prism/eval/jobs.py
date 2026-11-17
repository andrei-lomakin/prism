"""Filesystem-backed eval job state for UI progress (reports/.jobs/)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
JOBS_DIR = ROOT / "reports" / ".jobs"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_jobs_dir() -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def job_path(job_id: str) -> Path:
    return ensure_jobs_dir() / f"{job_id}.json"


def create_job(*, eval_config: str, skip_generation: bool) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    state = {
        "id": job_id,
        "status": "queued",
        "eval_config": eval_config,
        "skip_generation": bool(skip_generation),
        "step": "queued",
        "detail": "",
        "progress": 0,
        "i": 0,
        "n": 0,
        "question_id": None,
        "question": None,
        "run_id": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    save_job(state)
    return state


def save_job(state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = _now()
    path = job_path(str(state["id"]))
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_job(job_id: str) -> dict[str, Any] | None:
    path = job_path(job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    state = load_job(job_id)
    if state is None:
        return None
    state.update(fields)
    save_job(state)
    return state


def list_jobs(*, limit: int = 20) -> list[dict[str, Any]]:
    ensure_jobs_dir()
    rows: list[dict[str, Any]] = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, limit)]


# Human labels for diploma UI (stem → title / group / hint).
_EVAL_CONFIG_META: dict[str, dict[str, str]] = {
    "ablate_dense": {
        "title": "Векторный поиск (dense) · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Только эмбеддинги, без BM25. База для сравнения с hybrid.",
    },
    "ablate_hybrid": {
        "title": "Гибрид dense+BM25 (RRF) · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Смешанный поиск — обычно лучший Recall@5 на защите.",
    },
    "ablate_lexical": {
        "title": "Только слова (BM25) · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Классический лексический поиск без векторов.",
    },
    "ablate_minilm_l6": {
        "title": "Dense на MiniLM-L6 · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Другая модель эмбеддингов (английский уклон).",
    },
    "ablate_hybrid_test": {
        "title": "Гибрид · test (held-out)",
        "group": "Контрольный test",
        "hint": "Тот же hybrid на test.jsonl — не тюнить по этим цифрам.",
    },
    "baseline_test": {
        "title": "Baseline + ответы · test",
        "group": "Контрольный test",
        "hint": "Полный прогон на held-out; долго, если включена генерация.",
    },
    "baseline": {
        "title": "Baseline + ответы · dev",
        "group": "Полный прогон",
        "hint": "Поиск и генерация с LLM-judge на всём dev.",
    },
    "smoke": {
        "title": "Быстрая проверка (30 вопросов)",
        "group": "Диагностика",
        "hint": "Короткий retrieval-only прогон, чтобы убедиться что стек жив.",
    },
    "ci_smoke": {
        "title": "CI-мини (10 вопросов)",
        "group": "Диагностика",
        "hint": "Самый короткий срез для пайплайна CI.",
    },
}


def list_eval_configs() -> list[dict[str, str]]:
    cfg_dir = ROOT / "configs" / "eval"
    if not cfg_dir.is_dir():
        return []
    out: list[dict[str, str]] = []
    for path in sorted(cfg_dir.glob("*.yaml")):
        stem = path.stem
        rel = f"configs/eval/{path.name}"
        meta = _EVAL_CONFIG_META.get(stem) or {}
        out.append(
            {
                "id": rel,
                "name": stem,
                "path": rel,
                "title": meta.get("title") or stem,
                "group": meta.get("group") or "Прочее",
                "hint": meta.get("hint") or "",
            }
        )
    # Preferred order: comparison → test → full → diagnostic → other
    group_rank = {
        "Сравнение поиска (dev)": 0,
        "Контрольный test": 1,
        "Полный прогон": 2,
        "Диагностика": 3,
        "Прочее": 9,
    }
    out.sort(key=lambda c: (group_rank.get(c["group"], 8), c["title"]))
    return out
