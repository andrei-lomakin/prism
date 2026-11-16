"""Eval pipeline: golden dataset, retrieval/generation metrics, ablation runner.

Stage 1: schema + golden builder (`prism golden build`).
Stage 2: `prism eval run` + reports + LLM/heuristic judge.
"""

from prism.eval.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from prism.eval.runner import run_eval

__all__ = [
    "hit_rate_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "run_eval",
]
