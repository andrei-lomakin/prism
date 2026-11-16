"""Bootstrap CI helpers + paired comparison (Stage 7 / TZ §6.3)."""

from __future__ import annotations

import random
from typing import Any


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare mean(a)-mean(b). Returns delta and P(delta>0) under paired bootstrap."""
    if len(a) != len(b) or not a:
        return {
            "n": 0,
            "mean_a": None,
            "mean_b": None,
            "delta": None,
            "ci95_lo": None,
            "ci95_hi": None,
            "p_better": None,
        }
    n = len(a)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = sum(a[i] for i in idx) / n
        mb = sum(b[i] for i in idx) / n
        deltas.append(ma - mb)
    deltas.sort()
    lo = deltas[int(0.025 * (n_boot - 1))]
    hi = deltas[int(0.975 * (n_boot - 1))]
    p_better = sum(1 for d in deltas if d > 0) / n_boot
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    return {
        "n": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(mean_a - mean_b, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "p_better": round(p_better, 4),
    }


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    """Cohen's κ for categorical labels (same length)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b))
    # confusion
    idx = {c: i for i, c in enumerate(cats)}
    mat = [[0] * len(cats) for _ in cats]
    for x, y in zip(labels_a, labels_b, strict=True):
        mat[idx[x]][idx[y]] += 1
    po = sum(mat[i][i] for i in range(len(cats))) / n
    pe = 0.0
    for i in range(len(cats)):
        row = sum(mat[i][j] for j in range(len(cats))) / n
        col = sum(mat[j][i] for j in range(len(cats))) / n
        pe += row * col
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)
