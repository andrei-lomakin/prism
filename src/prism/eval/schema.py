"""Golden dataset schema (Stage 1).

Relevance is bound to stable `doc_key` + optional `section`, NOT chunk_id,
so the set survives re-chunking. Chunk-level relevance at eval time =
chunks whose text intersects the gold evidence span / section.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

QuestionType = Literal["factoid", "multi-hop", "comparison", "numeric", "unanswerable"]
Language = Literal["ru", "en"]
Split = Literal["dev", "test"]


class RelevantDoc(BaseModel):
    doc_key: str = Field(..., min_length=1)
    section: str | None = None


class GoldenQuestion(BaseModel):
    id: str
    question: str = Field(..., min_length=3)
    language: Language
    type: QuestionType
    relevant_docs: list[RelevantDoc] = Field(default_factory=list)
    gold_answer: str
    split: Split
    reviewed: bool = False
    notes: str = ""

    @field_validator("relevant_docs")
    @classmethod
    def unanswerable_has_no_docs(cls, v: list[RelevantDoc], info):
        # validated in model_validator-style below via type check in build
        return v

    def validate_consistency(self) -> None:
        if self.type == "unanswerable":
            if self.relevant_docs:
                raise ValueError(f"{self.id}: unanswerable must have empty relevant_docs")
        elif self.type == "multi-hop":
            keys = {d.doc_key for d in self.relevant_docs}
            if len(keys) < 2:
                raise ValueError(f"{self.id}: multi-hop needs ≥2 distinct doc_keys")
        elif not self.relevant_docs:
            raise ValueError(f"{self.id}: answerable types need relevant_docs")
