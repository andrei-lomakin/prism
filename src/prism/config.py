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
    llm_model: str = "qwen3:4b"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
