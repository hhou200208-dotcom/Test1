# Unified 25-Satellite 32K × Seed0 Benchmark

All five learning methods are retrained from scratch with the same frozen 25-satellite environment, workload, physical model, external reward definition, 32K physical-slot training budget, paired training seeds and fixed evaluation seeds.

Online convergence uses a short fixed-seed 64-slot deterministic evaluation every 640 training slots. The paper-facing final metrics are obtained from the frozen 32K state in an independent **5400-slot** deterministic evaluation.

## Frozen configuration

- N = 25 (5×5); hotspot ratio = 1/5.
- lambda_H = 4.0, lambda_L = 0.1, mean lambda = 0.88 task/slot/satellite.
- kappa = 1.5e-27; battery capacity = 54 kJ; DoD range [0, 0.8].
- V_DVFS = 2e17; solar max = 30 W; housekeeping = 5 W.
- External reward weights: done/timeout/reject/HL/queue = 10/5/5/2/0.05.
- PPO family: gamma=.99, GAE=.95, clip=.2, entropy=.02, actor LR=1e-4, critic LR=1e-3, epochs=2, rollout=64.
- BLA-MAPPO uses beta_task=.5; vanilla MAPPO/IPPO use beta_task=0.

## Final 5400-slot deterministic evaluation

| Algorithm | Return ↑ | Satisfaction ↑ | Delay (s/slot) ↓ | Energy (kJ/slot) ↓ | Cum lifetime loss ↓ |
|---|---:|---:|---:|---:|---:|
| BLA-MAPPO | 14820.236 | 0.4989 | 57.8407 | 0.032747 | 1.171638e+00 |
| MAPPO | 28773.147 | 0.7165 | 59.8596 | 0.047878 | 2.819562e+00 |
| IPPO | 21351.922 | 0.6105 | 58.8313 | 0.043819 | 2.483738e+00 |
| MADDPG | 11455.859 | 0.4491 | 52.5596 | 0.031054 | 1.549061e+00 |
| QMIX | 35329.281 | 0.8106 | 41.4589 | 0.045268 | 3.022124e+00 |

This is the **one-seed configuration acceptance run**. It validates the frozen experiment protocol but is not yet the final statistical claim. After acceptance, repeat the same protocol over 5 paired training seeds and report mean ± standard deviation (or 95% CI).
