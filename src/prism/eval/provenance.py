"""Stamp git commit / dirty flag into eval reports (Stage 7 reproducibility)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def git_provenance(cwd: Path | None = None) -> dict[str, Any]:
    root = cwd or ROOT
    out: dict[str, Any] = {
        "commit": None,
        "commit_short": None,
        "branch": None,
        "dirty": None,
        "describe": None,
    }
    try:
        commit = _run(["git", "rev-parse", "HEAD"], root)
        out["commit"] = commit
        out["commit_short"] = commit[:12] if commit else None
        out["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
        dirty = _run(["git", "status", "--porcelain"], root)
        out["dirty"] = bool(dirty.strip()) if dirty is not None else None
        out["describe"] = _run(
            ["git", "describe", "--always", "--dirty", "--tags"], root, check=False
        )
    except Exception:
        pass
    return out


def _run(cmd: list[str], cwd: Path, *, check: bool = True) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=check,
            timeout=5,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return None
