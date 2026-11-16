"""Mean metrics + simple bootstrap 95% CI."""

from __future__ import annotations

import random
from statistics import mean


def mean_or_zero(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for 1-alpha CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return float(mean(values)), float(lo), float(hi)


def summarize_metric(values: list[float]) -> dict:
    m, lo, hi = bootstrap_ci(values)
    return {
        "mean": round(m, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "n": len(values),
    }
