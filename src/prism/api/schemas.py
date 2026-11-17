"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class SettingsOut(BaseModel):
    embedding_model: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_mode: str
    vector_search: Literal["exact", "hnsw"] = "exact"
    hnsw_ef_search: int = 40
    rerank_enabled: bool = False
    rerank_candidates: int = 20
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    space_default: str


class SettingsUpdate(BaseModel):
    embedding_model: str | None = None
    chunking_strategy: str | None = None
    chunk_size: int | None = Field(default=None, ge=32, le=4096)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1024)
    top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_mode: Literal["dense", "lexical", "hybrid"] | None = None
    vector_search: Literal["exact", "hnsw"] | None = None
    hnsw_ef_search: int | None = Field(default=None, ge=10, le=400)
    rerank_enabled: bool | None = None
    rerank_candidates: int | None = Field(default=None, ge=5, le=100)
    llm_provider: Literal["ollama", "openai_compatible"] | None = None
    llm_model: str | None = None
    ollama_base_url: str | None = None
    space_default: Literal["personal", "project", "shared"] | None = None


class UrlIngestRequest(BaseModel):
    url: HttpUrl
    space: Literal["personal", "project", "shared"] = "personal"


class UploadResponse(BaseModel):
    document_id: int
    job_id: int


class DocumentOut(BaseModel):
    id: int
    name: str
    source_type: str
    source_path: str
    source_url: str | None = None
    space: str
    status: str
    num_chunks: int
    error_message: str | None = None
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    items: list[DocumentOut]
    total: int
    page: int
    page_size: int
    pages: int


class JobOut(BaseModel):
    id: int
    document_id: int
    kind: str
    status: str
    progress: int
    current_step: str | None = None
    error_message: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    space: Literal["all", "personal", "project", "shared"] = "all"


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    document_name: str
    text: str
    score: float
    section_path: str | None = None
    page: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    retrieval_mode: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["rag", "agent"] = "rag"
    space: Literal["all", "personal", "project", "shared"] = "all"


class Citation(BaseModel):
    n: int
    chunk_id: int | None = None
    doc_id: int | None = None
    title: str
    section_path: str | None = None
    page: int | None = None
    score: float
    text: str | None = None  # chunk / web snippet for UI highlight
    url: str | None = None
    kind: Literal["kb", "web"] | None = "kb"


class UsageOut(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


class GroundingOut(BaseModel):
    ok: bool
    cited: list[int] = Field(default_factory=list)
    missing: list[int] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    unsupported_names: list[str] = Field(default_factory=list)
    requires_citation: bool = False


class ChatResponse(BaseModel):
    query: str
    answer: str
    model: str
    citations: list[Citation]
    retrieval_mode: str
    grounding_ok: bool
    grounding: GroundingOut | None = None
    usage: UsageOut
    trace_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    detail: dict[str, Any] = Field(default_factory=dict)
