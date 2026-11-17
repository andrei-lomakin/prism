| run_id | role | retrieval | chunking | n_q | Recall@5 | MRR@10 | nDCG@10 | Hit@5 | Faith. | AnsRel | p95_ms | $/1k | commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261116T105854Z_smoke | diagnostic | dense | C2 | 30 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | 1593.2 | 0.0000 | — |
| 20261116T110301Z_smoke | diagnostic | dense | C2 | 30 | 0.925 | 0.668 | 0.760 | 0.950 | — | — | 1336.0 | 0.0000 | — |
| 20261116T132843Z_ablate_hybr | diagnostic | hybrid | C2 | 60 | 0.934 | 0.754 | 0.817 | 1.000 | — | — | 2083.8 | 0.0000 | — |
| 20261117T110032Z_ablate_dens | diagnostic | dense | C2 | 60 | 0.590 | 0.355 | 0.407 | 0.667 | — | — | 33.7 | 0.0000 | 29bd526c2332 |
| 20261117T110132Z_ablate_hybr | scientific | hybrid | C2 | 60 | 0.808 | 0.695 | 0.695 | 0.949 | — | — | 42.3 | 0.0000 | 29bd526c2332 |
| 20261117T110158Z_ablate_dens | scientific | dense | C2 | 60 | 0.590 | 0.355 | 0.407 | 0.667 | — | — | 118.7 | 0.0000 | 29bd526c2332 |
| 20261117T110614Z_ablate_hybr | scientific | hybrid | C2 | 66 | 0.868 | 0.669 | 0.722 | 0.947 | — | — | 431.1 | 0.0000 | 29bd526c2332 |
| 20261117T110652Z_baseline_te | scientific | dense | C2 | 66 | 0.412 | 0.344 | 0.403 | 0.421 | — | — | 414.4 | 0.0000 | 29bd526c2332 |

## Best vs baseline (paired bootstrap on recall@5)

Baseline: explicit `--baseline` / `--baseline-config`, else `ablate_dense` -> `baseline.yaml` (smoke/diagnostic excluded).
Pairs join on `question_id`; CI requires comparable controls + >=2 shared ids.

```json
{
  "best_run": "20261117T110132Z_ablate_hybrid",
  "baseline_run": "20261117T110158Z_ablate_dense",
  "baseline_eval_config": "configs/eval/ablate_dense.yaml",
  "metric": "recall@5",
  "comparability": {
    "ok": true,
    "mismatches": []
  },
  "n_paired_questions": 39,
  "n_only_best": 0,
  "n_only_baseline": 0,
  "paired_bootstrap": {
    "n": 39,
    "mean_a": 0.8077,
    "mean_b": 0.5897,
    "delta": 0.2179,
    "ci95_lo": 0.1154,
    "ci95_hi": 0.3205,
    "p_better": 1.0
  },
  "sample_question_ids": [
    "para_m_agent_budget_en_029",
    "para_m_agent_budget_en_034",
    "para_m_ground_eval_en_025",
    "para_m_hybrid_packer_en_022",
    "para_m_hybrid_packer_en_027"
  ]
}
```
