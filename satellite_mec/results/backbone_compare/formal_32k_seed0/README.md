# Formal 32K × 1 Seed Experiment

This directory contains the complete one-seed 32K run for **BLA-MAPPO, MAPPO, IPPO, MADDPG and QMIX**. It is a full engineering run, but **not yet the final statistical claim** because only seed 0 is included.

## Unified metrics

Both training rollouts and deterministic evaluation checkpoints record return, completion rate, E2E delay, compute+transmit energy, cumulative battery health loss, and DoD. Evaluation uses the same fixed seeds for all five algorithms.

## Final deterministic evaluation checkpoint

| Algorithm | Return ↑ | CR ↑ | Delay (s) ↓ | Energy (J) ↓ | Cum-HL (mean sat.) ↓ |
|---|---:|---:|---:|---:|---:|
| BLA-MAPPO | -249.083 | 0.6670 | 4.5816 | 6847.787 | 3.678615e-03 |
| MAPPO | -137.075 | 0.7304 | 3.3531 | 5843.183 | 2.764244e-03 |
| IPPO | -379.695 | 0.7156 | 4.0724 | 7548.802 | 5.142076e-03 |
| MADDPG | -349.264 | 0.6311 | 4.8576 | 9353.988 | 4.561986e-03 |
| QMIX | -231.801 | 0.6702 | 4.7366 | 6247.736 | 3.521547e-03 |

## Files

- `<ALGORITHM>/seed0/result.json`: complete raw experiment record.
- `<ALGORITHM>/seed0/train_rollouts.csv`: 64-slot training metrics.
- `<ALGORITHM>/seed0/eval_checkpoints.csv`: fixed-seed deterministic evaluation metrics.
- `eval_checkpoints_all.csv`: plotting-ready combined evaluation table.
- `train_rollouts_all.csv`: plotting-ready combined training table.
- `summary_seed0.csv`: compact final-checkpoint table.
- `figures/`: reward/CR/delay/energy/HL curves.

**Interpretation:** one seed can reveal implementation failures and broad trends but must not be used to claim statistical superiority. The final paper comparison should use the same frozen protocol over 5 paired training seeds and report mean ± standard deviation.
