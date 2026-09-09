# MARL Backbone Convergence Experiment

> Branch: `experiment/marl-backbone-smoke`
>
> Purpose: compare **MAPPO / IPPO / MADDPG / QMIX** as reinforcement-learning backbones for the satellite MEC task-offloading problem under one controlled protocol.

This branch is an **experiment branch**, not the final paper result branch. It currently contains smoke tests, fairness fixes, hyper-parameter pilot runs, and the first **8K-slot / seed-0 pilot**. The final result will only be reported after the planned **32K slots × 5 paired seeds** experiment.

## Start here

```text
Test1/
├── README.md                              <- you are here
├── .github/workflows/                     <- cloud experiment workflows
│   ├── marl-real-env-smoke.yml
│   ├── backbone-tune.yml
│   ├── maddpg-tune.yml
│   ├── marl-mini-pilot.yml
│   └── marl-pilot-8k.yml
└── satellite_mec/
    ├── backbone/                          <- experiment implementation
    │   ├── README.md                      <- code guide
    │   ├── compare_algorithms.py          <- MAPPO/IPPO/MADDPG/QMIX adapters
    │   ├── fairness_fixes.py              <- controlled-comparison fixes
    │   ├── real_env_smoke.py              <- real SatelliteMECEnv smoke test
    │   ├── backbone_tune.py               <- MAPPO/IPPO/QMIX tuning pilot
    │   ├── maddpg_tune.py                 <- MADDPG tuning pilot
    │   ├── pilot_compare.py               <- short common-protocol pilot
    │   └── pilot_8k.py                    <- frozen-parameter 8K pilot
    └── results/backbone_compare/          <- committed experiment records
        ├── README.md                      <- result/status guide
        ├── smoke/
        ├── tuning/
        └── pilot_8k/                      <- 8K seed-0 raw JSON + summary
```

## Controlled comparison protocol

The experiment is designed to isolate the RL backbone as much as possible:

- Same `SatelliteMECEnv` and same physical/task configuration.
- Same dynamic action feasibility mask.
- Same cooperative system reward: `R_sys = mean_n(r_n)`.
- PPO task-level credit refinement is disabled for backbone selection: `BETA_TASK = 0`.
- MAPPO and IPPO share the same PPO actor/update settings; the main intended difference is critic information.
- MADDPG uses the repository's discrete Gumbel-Softmax actor-critic implementation, but the comparison replaces its special baseline reward with the common `R_sys`.
- QMIX uses a decision-round micro-step adapter with an internal NOOP for satellites without a task in a given round.
- Training budget is compared in **physical environment slots**, not gradient-update count.
- Evaluation uses a separate fixed-seed environment with exploration disabled.

## Current status

| Stage | Status | Notes |
|---|---|---|
| Tensor-level algorithm smoke test | ✅ done | Four algorithms pass forward/backward/mask checks |
| Real `SatelliteMECEnv` smoke test | ✅ done | Four algorithms can train/update in the real environment |
| Fairness fixes | ✅ done | Shared reward, PPO task credit off, IPPO critic isolation, MADDPG reward handling |
| Small tuning pilot | ✅ done | Parameters frozen before the 8K run |
| 8K slots × seed 0 | ✅ done | Raw JSON committed under `pilot_8k/` |
| 32K slots × 5 paired seeds | ⏳ not run yet | This will be the basis of the final convergence figure |
| mean ± std convergence figure | ⏳ not final yet | Do not use seed-0 pilot as the final paper figure |

## Important interpretation warning

The current 8K result is **one seed only**. It is useful for engineering validation and for deciding whether the formal experiment is worth running, but it is not statistically sufficient to claim that one backbone is superior. The final answer to “why MAPPO?” must be based on the multi-seed experiment, including mean return, variance/stability, sample efficiency/AUC, and convergence speed.

For code details, open `satellite_mec/backbone/README.md`. For raw data and current numerical summaries, open `satellite_mec/results/backbone_compare/README.md`.
