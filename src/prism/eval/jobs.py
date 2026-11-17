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
    reap_stale_jobs()
    rows: list[dict[str, Any]] = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[: max(1, limit)]


def reap_stale_jobs(*, queued_max_sec: int = 180, running_max_sec: int = 7200) -> int:
    """Mark zombie jobs failed (e.g. worker missed function / container FS split)."""
    ensure_jobs_dir()
    now = datetime.now(UTC)
    n = 0
    for path in JOBS_DIR.glob("*.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = state.get("status")
        if status not in ("queued", "running"):
            continue
        raw = state.get("updated_at") or state.get("created_at") or ""
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except Exception:
            continue
        age = (now - ts).total_seconds()
        limit = queued_max_sec if status == "queued" else running_max_sec
        if age < limit:
            continue
        state["status"] = "failed"
        state["step"] = "stale"
        state["error"] = (
            "Прогон завис: worker не подхватил задачу (перезапустите worker, "
            "убедитесь что смонтирован ./reports). Можно запустить снова."
        )
        state["detail"] = state["error"]
        save_job(state)
        n += 1
    return n


# Human labels for diploma UI (stem → title / group / hint).
_EVAL_CONFIG_META: dict[str, dict[str, str]] = {
    "ablate_dense": {
        "title": "Векторный поиск (dense) · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Только dense-retrieval по эмбеддингам, без BM25. Контроль для сравнения с hybrid.",
    },
    "ablate_hybrid": {
        "title": "Гибрид dense+BM25 (RRF) · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Dense + BM25, слияние через RRF. Обычно выше Recall@5, чем у чистого dense.",
    },
    "ablate_lexical": {
        "title": "Только слова (BM25) · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Только лексический BM25, без векторного поиска.",
    },
    "ablate_minilm_l6": {
        "title": "Dense на MiniLM-L6 · dev",
        "group": "Сравнение поиска (dev)",
        "hint": "Dense с другой моделью эмбеддингов (all-MiniLM-L6-v2).",
    },
    "ablate_hybrid_test": {
        "title": "Гибрид · test (held-out)",
        "group": "Контрольный test",
        "hint": "Hybrid на test.jsonl. Held-out: не подбирать гиперпараметры по этим цифрам.",
    },
    "baseline_test": {
        "title": "Baseline + ответы · test",
        "group": "Контрольный test",
        "hint": "Полный пайплайн на held-out test; с генерацией занимает заметно больше времени.",
    },
    "baseline": {
        "title": "Baseline + ответы · dev",
        "group": "Полный прогон",
        "hint": "Retrieval + генерация ответов и LLM-judge на dev-сплите.",
    },
    "smoke": {
        "title": "Быстрая проверка (30 вопросов)",
        "group": "Диагностика",
        "hint": "Короткий retrieval-only прогон для проверки, что индекс и retrieve живы.",
    },
    "ci_smoke": {
        "title": "CI-мини (10 вопросов)",
        "group": "Диагностика",
        "hint": "Минимальный срез для CI / smoke в пайплайне.",
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
