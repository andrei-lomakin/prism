"""BM25 lexical index (bm25s), persisted under BM25_DIR.

One corpus snapshot per collection name. Built after ingest; searched by
retrieve() in lexical/hybrid modes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Light RU/EN tokenizer: lowercase alphanumerics. Stemming optional later."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self, path: Path, *, collection: str = "default") -> None:
        self.path = Path(path)
        self.collection = collection
        self._ids: list[str] = []
        self._retriever = None
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def _ids_file(self) -> Path:
        return self.path / f"{self.collection}.ids.json"

    @property
    def _index_dir(self) -> Path:
        return self.path / f"{self.collection}.bm25"

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts length mismatch")
        if not doc_ids:
            self._ids = []
            self._retriever = None
            return

        import bm25s

        corpus_tokens = [tokenize(t) for t in texts]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self._index_dir))
        self._ids_file.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
        self._ids = list(doc_ids)
        self._retriever = retriever
        logger.info("BM25 built collection=%s docs=%s", self.collection, len(doc_ids))

    def load(self) -> bool:
        if not self._ids_file.exists() or not self._index_dir.exists():
            return False
        import bm25s

        self._ids = json.loads(self._ids_file.read_text(encoding="utf-8"))
        self._retriever = bm25s.BM25.load(str(self._index_dir), load_corpus=False)
        return True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._retriever is None and not self.load():
            return []
        if not self._ids:
            return []
        import bm25s
        import numpy as np

        tokens = tokenize(query)
        if not tokens:
            return []
        query_tokens = bm25s.tokenize([query])
        # bm25s returns (docs, scores) shaped arrays
        docs, scores = self._retriever.retrieve(query_tokens, k=min(top_k, len(self._ids)))
        # docs[0] are indices into corpus
        idxs = np.asarray(docs[0]).tolist()
        sc = np.asarray(scores[0]).tolist()
        out: list[tuple[str, float]] = []
        for i, s in zip(idxs, sc):
            if i is None or int(i) < 0 or int(i) >= len(self._ids):
                continue
            out.append((self._ids[int(i)], float(s)))
        return out
