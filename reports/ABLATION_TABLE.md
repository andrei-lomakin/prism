| run_id | retrieval | chunking | Recall@5 | MRR@10 | nDCG@10 | Hit@5 | Faith. | AnsRel | p95_ms | $/1k | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261116T105854Z_smoke | dense | C2 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 1593.2 | 0.0000 | — |
| 20261116T110301Z_smoke | dense | C2 | 0.925 | 0.668 | 0.760 | 0.950 | — | — | 1336.0 | 0.0000 | — |
| 20261116T132843Z_ablate_hybr | hybrid | C2 | 0.934 | 0.754 | 0.817 | 1.000 | — | — | 2083.8 | 0.0000 | — |

## Best vs baseline (paired bootstrap on recall@5)

```json
{
  "best_run": "20261116T132843Z_ablate_hybrid",
  "baseline_run": "20261116T105854Z_smoke",
  "metric": "recall@5",
  "paired_bootstrap": {
    "n": 20,
    "mean_a": 0.975,
    "mean_b": 0.0,
    "delta": 0.975,
    "ci95_lo": 0.925,
    "ci95_hi": 1.0,
    "p_better": 1.0
  }
}
```
