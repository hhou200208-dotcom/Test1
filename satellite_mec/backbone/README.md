# Backbone Experiment Code Guide

This directory contains the controlled RL-backbone comparison used to answer the course-project question: **why choose MAPPO instead of another multi-agent RL method?**

## Files

| File | Role |
|---|---|
| `compare_algorithms.py` | Common comparison adapters for MAPPO, IPPO, MADDPG and QMIX |
| `fairness_fixes.py` | Fixes needed to keep reward/critic conventions comparable |
| `smoke_test.py` | Tensor-level unit smoke test |
| `real_env_smoke.py` | Short test in the real `SatelliteMECEnv` |
| `backbone_tune.py` | Equal-budget small tuning run for MAPPO/IPPO/QMIX |
| `maddpg_tune.py` | Equal-budget small tuning run for MADDPG |
| `pilot_compare.py` | Common-protocol short pilot runner |
| `pilot_8k.py` | Frozen-hyperparameter 8K-slot seed-0 pilot |

## What is being compared

The comparison intentionally uses **vanilla backbone variants**, not the full proposed BLA-MAPPO method.

- `BETA_TASK = 0` disables the BLA task-level credit refinement during backbone selection.
- All four methods receive the same cooperative system reward `R_sys`.
- Dynamic action masks are retained for every method.
- MAPPO and IPPO use the same PPO actor/update mechanism. IPPO removes the centralized critic information so that MAPPO-vs-IPPO is a controlled CTDE comparison.
- MADDPG uses a discrete Gumbel-Softmax actor because the task-offloading action is discrete.
- QMIX groups the k-th task decision of all satellites in a slot into a decision round and uses an internal NOOP for inactive satellites.

## Formal experiment protocol planned next

The final convergence experiment should freeze the current algorithm definitions and hyperparameters, then run:

- 4 algorithms: MAPPO / IPPO / MADDPG / QMIX
- 5 paired random seeds
- 32K physical training slots per run
- fixed deterministic evaluation checkpoints
- final figure: mean evaluation episode return ± standard deviation
- summary metrics: final return, AUC, convergence threshold/T90, final standard deviation

Do not tune hyperparameters after observing the 5-seed formal results.

## Reproducibility note

Cloud runs are defined in `.github/workflows/`. The current 8K run was executed by `marl-pilot-8k.yml`. Its committed raw result files are under `satellite_mec/results/backbone_compare/pilot_8k/`.
