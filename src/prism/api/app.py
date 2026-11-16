"""FastAPI entrypoint: JSON API + simple Jinja pages."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from prism.api.routes import chat, documents, eval_ui, memory, runs
from prism.api.routes import settings as settings_routes
from prism.api.settings_store import get_effective_settings
from prism.config import get_settings
from prism.db.session import get_db, init_db
from prism.observability import setup_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = get_settings()
    setup_tracing(
        cfg.otel_service_name, enabled=cfg.otel_enabled, endpoint=cfg.otel_exporter_endpoint
    )
    try:
        init_db()
        logger.info("Database ready")
    except Exception:
        logger.exception("DB init failed — start postgres first")
    yield


app = FastAPI(
    title="Prism",
    description="Personal research assistant: RAG + multi-agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(settings_routes.router)
app.include_router(runs.router)
app.include_router(eval_ui.router)
app.include_router(memory.router)


@app.get("/", response_class=RedirectResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/search", status_code=302)


@app.get("/search", response_class=HTMLResponse)
def page_search(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    s = get_effective_settings(db)
    return templates.TemplateResponse(
        request,
        "search.html",
        {"active": "search", "default_top_k": s.top_k},
    )


@app.get("/upload", response_class=HTMLResponse)
def page_upload(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "upload.html", {"active": "upload"})


@app.get("/settings", response_class=HTMLResponse)
def page_settings(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"active": "settings", "settings": get_effective_settings(db)},
    )


@app.get("/trace", response_class=HTMLResponse)
def page_trace(
    request: Request, run_id: str | None = None, trace_id: str | None = None
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "trace.html",
        {"active": "trace", "run_id": run_id or "", "trace_id": trace_id or ""},
    )


@app.get("/approvals", response_class=HTMLResponse)
def page_approvals(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "approvals.html", {"active": "approvals"})


@app.get("/eval", response_class=HTMLResponse)
def page_eval(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "eval.html", {"active": "eval"})
