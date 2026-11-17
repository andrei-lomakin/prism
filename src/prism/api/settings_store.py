"""Persist UI-editable settings overlay in Postgres (AppSetting id=1)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from prism.api.schemas import SettingsOut, SettingsUpdate
from prism.config import Settings, get_settings
from prism.db.models import AppSetting


def _base_out(s: Settings) -> SettingsOut:
    return SettingsOut(
        embedding_model=s.embedding_model,
        chunking_strategy=s.chunking_strategy,
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        top_k=s.top_k,
        retrieval_mode=s.retrieval_mode,
        vector_search=s.vector_search,  # type: ignore[arg-type]
        hnsw_ef_search=int(s.hnsw_ef_search),
        rerank_enabled=bool(s.rerank_enabled),
        rerank_candidates=int(s.rerank_candidates),
        rerank_model=s.rerank_model,
        llm_provider=s.llm_provider,
        llm_model=s.llm_model,
        ollama_base_url=s.ollama_base_url,
        space_default=s.space_default,
    )


def get_effective_settings(db: Session) -> SettingsOut:
    row = db.get(AppSetting, 1)
    base = _base_out(get_settings())
    if row is None or not row.payload:
        return base
    data = base.model_dump()
    data.update({k: v for k, v in row.payload.items() if v is not None and k in data})
    return SettingsOut(**data)


def save_settings(db: Session, update: SettingsUpdate) -> SettingsOut:
    row = db.get(AppSetting, 1)
    current = get_effective_settings(db).model_dump()
    prev_emb = current.get("embedding_model")
    patch = update.model_dump(exclude_unset=True)
    current.update(patch)
    if row is None:
        row = AppSetting(id=1, payload=current)
        db.add(row)
    else:
        row.payload = current
    db.commit()
    if patch.get("embedding_model") and patch["embedding_model"] != prev_emb:
        from prism.rag.embeddings.base import clear_embedder_cache

        clear_embedder_cache()
    llm_keys = ("llm_provider", "llm_model", "ollama_base_url")
    if any(k in patch for k in llm_keys):
        from prism.llm.gateway import clear_llm_cache

        clear_llm_cache()
    if "vector_search" in patch:
        from prism.db.session import get_engine
        from prism.rag.index.vector_ann import ensure_vector_index

        mode = str(current.get("vector_search") or "exact")
        if mode not in ("exact", "hnsw"):
            mode = "exact"
        ensure_vector_index(get_engine(), mode)  # type: ignore[arg-type]
    return get_effective_settings(db)


def settings_as_config(db: Session) -> Settings:
    """Merge env defaults with DB overlay into a Settings object for RAG calls."""
    out = get_effective_settings(db)
    base = get_settings()
    return base.model_copy(
        update={
            "embedding_model": out.embedding_model,
            "chunking_strategy": out.chunking_strategy,
            "chunk_size": out.chunk_size,
            "chunk_overlap": out.chunk_overlap,
            "top_k": out.top_k,
            "retrieval_mode": out.retrieval_mode,
            "vector_search": out.vector_search,
            "hnsw_ef_search": int(out.hnsw_ef_search),
            "rerank_enabled": bool(out.rerank_enabled),
            "rerank_candidates": int(out.rerank_candidates),
            "llm_provider": out.llm_provider,
            "llm_model": out.llm_model,
            "ollama_base_url": out.ollama_base_url,
            "space_default": out.space_default,
        }
    )
