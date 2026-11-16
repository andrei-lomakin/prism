"""Embedding interface.

All models share embed_documents / embed_query so ablations swap cleanly.
Hash cache avoids re-embedding identical strings within a process.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from functools import lru_cache

logger = logging.getLogger(__name__)


class Embedder(ABC):
    model_name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic fake embedder for unit tests / CI without model download."""

    def __init__(self, dim: int = 384, model_name: str = "hash-embedder") -> None:
        self.dim = dim
        self.model_name = model_name

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = []
        while len(vals) < self.dim:
            for b in digest:
                vals.append((b / 255.0) * 2 - 1)
                if len(vals) >= self.dim:
                    break
            digest = hashlib.sha256(digest).digest()
        # L2-normalize
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dim_fn = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = int(dim_fn())
        self._cache: dict[str, list[float]] = {}
        self._needs_e5_prefix = "e5" in model_name.lower()

    def _key(self, text: str, kind: str) -> str:
        return hashlib.sha256(f"{self.model_name}:{kind}:{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            k = self._key(t, "doc")
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                payload = f"passage: {t}" if self._needs_e5_prefix else t
                missing.append((i, payload))
        if missing:
            vectors = self._model.encode(
                [t for _, t in missing],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for (i, _), vec in zip(missing, vectors, strict=True):
                arr = vec.tolist()
                self._cache[self._key(texts[i], "doc")] = arr
                out[i] = arr
        return [v for v in out if v is not None]  # type: ignore[misc]

    def embed_query(self, text: str) -> list[float]:
        k = self._key(text, "query")
        if k in self._cache:
            return self._cache[k]
        payload = f"query: {text}" if self._needs_e5_prefix else text
        vec = self._model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
        arr = vec.tolist()
        self._cache[k] = arr
        return arr


@lru_cache
def get_embedder(model_name: str | None = None) -> Embedder:
    from prism.config import get_settings

    name = model_name or get_settings().embedding_model
    if name in ("hash", "hash-embedder") or name.startswith("hash:"):
        return HashEmbedder()
    return SentenceTransformerEmbedder(name)


def clear_embedder_cache() -> None:
    get_embedder.cache_clear()
