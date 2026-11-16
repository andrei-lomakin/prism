"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class SettingsOut(BaseModel):
    embedding_model: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_mode: str
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    space_default: str


class SettingsUpdate(BaseModel):
    embedding_model: Optional[str] = None
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=32, le=4096)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1024)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    retrieval_mode: Optional[Literal["dense", "lexical", "hybrid"]] = None
    llm_provider: Optional[Literal["ollama", "openai_compatible"]] = None
    llm_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    space_default: Optional[Literal["personal", "project", "shared"]] = None


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
    source_url: Optional[str] = None
    space: str
    status: str
    num_chunks: int
    error_message: Optional[str] = None
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
    current_step: Optional[str] = None
    error_message: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    document_name: str
    text: str
    score: float
    section_path: Optional[str] = None
    page: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    retrieval_mode: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    mode: Literal["rag", "agent"] = "rag"


class Citation(BaseModel):
    n: int
    chunk_id: int
    doc_id: int
    title: str
    section_path: Optional[str] = None
    page: Optional[int] = None
    score: float


class UsageOut(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


class ChatResponse(BaseModel):
    query: str
    answer: str
    model: str
    citations: list[Citation]
    retrieval_mode: str
    grounding_ok: bool
    usage: UsageOut
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    detail: dict[str, Any] = Field(default_factory=dict)
