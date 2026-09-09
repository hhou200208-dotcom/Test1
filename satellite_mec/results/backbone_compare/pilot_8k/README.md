# 8K-slot Pilot Results (Seed 0)

> **Pilot only — not the final convergence result.** This directory contains the first frozen-hyperparameter run using 8,192 physical training slots and one paired environment seed. Do not use this one-seed result to claim algorithm superiority.

## Files

- `MAPPO/seed0.json` — raw checkpoint record
- `IPPO/seed0.json` — raw checkpoint record
- `MADDPG/seed0.json` — raw checkpoint record
- `QMIX/seed0.json` — raw checkpoint record
- `summary_seed0.csv` / `.json` — compact final metrics
- `curves_seed0.csv` — plotting-ready checkpoint table

## Final checkpoint at 8K slots

| Algorithm | Eval episode return ↑ | Completion rate ↑ |
|---|---:|---:|
| MAPPO | -473.24 | 0.821 |
| IPPO | -322.45 | 0.747 |
| MADDPG | -408.18 | 0.685 |
| QMIX | -279.30 | 0.659 |

The last checkpoint is noisy: the curves fluctuate substantially within a single seed. The formal experiment therefore remains **32K slots × 5 paired seeds**, reported as mean ± standard deviation with Final Return, AUC, T90 and final-run variance.
