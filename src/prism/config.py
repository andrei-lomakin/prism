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

    database_url: str = "postgresql+psycopg://prism:prism@localhost:5432/prism"
    redis_url: str = "redis://localhost:6379/0"

    llm_provider: Literal["ollama", "openai_compatible"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:latest"
    openai_api_base: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Known dims for default models; Embedder reports actual dim at load time.
    embedding_dim: int = 384
    chunking_strategy: ChunkStrategy = "C2"
    chunk_size: int = Field(default=256, ge=32, le=4096)
    chunk_overlap: int = Field(default=32, ge=0, le=1024)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_mode: RetrievalMode = "dense"
    space_default: SpaceName = "personal"

    upload_dir: Path = Path("./data/uploads")
    bm25_dir: Path = Path("./data/bm25")

    # Context packing (S2-style budget)
    pack_max_chunks: int = 8
    pack_max_tokens: int = 2048
    pack_max_per_doc: int = 3
    pack_min_score: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
