"""Settings + health + Ollama model list."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from prism import __version__
from prism.api.schemas import HealthResponse, SettingsOut, SettingsUpdate
from prism.api.settings_store import get_effective_settings, save_settings, settings_as_config
from prism.db.session import get_db
from prism.llm import LLMGateway

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/settings", response_model=SettingsOut)
def api_get_settings(db: Session = Depends(get_db)) -> SettingsOut:
    return get_effective_settings(db)


@router.post("/settings", response_model=SettingsOut)
def api_update_settings(payload: SettingsUpdate, db: Session = Depends(get_db)) -> SettingsOut:
    return save_settings(db, payload)


@router.get("/ollama/models")
def api_ollama_models(db: Session = Depends(get_db)) -> dict:
    cfg = settings_as_config(db)
    gateway = LLMGateway(cfg)
    try:
        models = gateway.list_models()
    except Exception as exc:
        raise HTTPException(503, f"Ollama unavailable: {exc}") from exc
    return {"models": models, "ollama_base_url": cfg.ollama_base_url}
