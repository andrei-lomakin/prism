"""Application settings from environment / .env.

Single source of truth for runtime knobs. RAG ablation configs live in
`configs/rag/*.yaml` and override these defaults for eval runs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkStrategy = Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
RetrievalMode = Literal["dense", "lexical", "hybrid"]
HybridFusion = Literal["rrf", "weighted"]
RewriteStrategy = Literal["R0", "R1", "R2", "R3", "R4", "R5"]
SpaceName = Literal["personal", "project", "shared"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    prism_env: str = "dev"
    prism_secret_key: str = "change-me"
    prism_log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://prism:prism@127.0.0.1:5434/prism"
    redis_url: str = "redis://localhost:6379/0"

    llm_provider: Literal["ollama", "openai_compatible"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    openai_api_base: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    chunking_strategy: ChunkStrategy = "C2"
    chunk_size: int = Field(default=256, ge=32, le=4096)
    chunk_overlap: int = Field(default=32, ge=0, le=1024)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_mode: RetrievalMode = "dense"
    space_default: SpaceName = "personal"
    # Runtime filter from chat UI (None = all spaces). Not a durable .env knob.
    space_filter: str | None = None

    # Hybrid / rewrite / rerank (Stage 3)
    hybrid_fusion: HybridFusion = "rrf"
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rewrite_strategy: RewriteStrategy = "R0"
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidates: int = Field(default=20, ge=5, le=100)

    upload_dir: Path = Path("./data/uploads")
    bm25_dir: Path = Path("./data/bm25")

    pack_max_chunks: int = 8
    pack_max_tokens: int = 2048
    pack_max_per_doc: int = 3
    pack_min_score: float = 0.0

    # S1 space hierarchy weights (personal > project > shared)
    space_weight_personal: float = Field(default=1.0, ge=0.0, le=2.0)
    space_weight_project: float = Field(default=0.85, ge=0.0, le=2.0)
    space_weight_shared: float = Field(default=0.72, ge=0.0, le=2.0)
    space_weights_enabled: bool = True

    # LLMGateway extras (S7)
    llm_cache_enabled: bool = True
    llm_pii_redact: bool = True
    llm_input_usd_per_1m: float = 0.0  # local Ollama default = free
    llm_output_usd_per_1m: float = 0.0
    # Comma-separated fallback models tried after primary fails (Ollama tags)
    llm_fallback_models: str = ""

    # Optional single-user auth (empty password = disabled)
    auth_username: str = "prism"
    auth_password: str = ""

    # ActionGate approval TTL (seconds); 0 = no expiry
    approval_ttl_seconds: int = Field(default=300, ge=0, le=86400)

    # Observability (Stage 5)
    otel_enabled: bool = False
    otel_service_name: str = "prism"
    otel_exporter_endpoint: str | None = None  # e.g. http://localhost:4317

    # Stage 6 web tools (mock by default for offline demos)
    web_search_mock: bool = True
    distill_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
