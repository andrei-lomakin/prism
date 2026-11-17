| run_id | role | retrieval | chunking | n_q | Recall@5 | MRR@10 | nDCG@10 | Hit@5 | Faith. | AnsRel | p95_ms | $/1k | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261116T105854Z_smoke | diagnostic | dense | C2 | 30 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 1593.2 | 0.0000 | — |
| 20261116T110301Z_smoke | diagnostic | dense | C2 | 30 | 0.925 | 0.668 | 0.760 | 0.950 | — | — | 1336.0 | 0.0000 | — |
| 20261116T132843Z_ablate_hybr | scientific | hybrid | C2 | 60 | 0.934 | 0.754 | 0.817 | 1.000 | — | — | 2083.8 | 0.0000 | — |

## Best vs baseline (paired bootstrap on recall@5)

Baseline: explicit `--baseline` / `--baseline-config`, else `ablate_dense` -> `baseline.yaml` (smoke/diagnostic excluded).
Pairs join on `question_id`; CI requires comparable controls + >=2 shared ids.

```json
{
  "error": "no_scientific_baseline",
  "hint": "Pass --baseline RUN_ID or run configs/eval/ablate_dense.yaml / baseline.yaml"
}
```
