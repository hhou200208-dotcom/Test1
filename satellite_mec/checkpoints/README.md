# Trained Model Checkpoints

Saved trained model weights for paper-grade reproducibility.
These are tracked by git (unlike `results/` which is gitignored).

> ## ⚠️ CRITICAL — `LyaMAPPO_lh4_32K` is an IRREPRODUCIBLE golden checkpoint
> This is the paper's headline model (satisfaction/CR ≈ **0.786**, HL/slot **1.74e-4**).
> M15 (2026-06-21) found training-seed variance is large: a **fresh** V=50@32K
> retrain with identical hyperparameters reached only **CR 0.745 / HL 2.89e-4**
> (−4.1 pp / +66 %). **This exact checkpoint cannot be reliably reproduced by
> retraining.** Therefore:
> - **NEVER delete, overwrite, or re-train into this directory.**
> - It is committed to git AND pushed to origin (survives container loss).
> - Integrity is pinned in `LyaMAPPO_lh4_32K/CHECKSUMS.md5` (verify with
>   `md5sum -c CHECKSUMS.md5`).
> - Marked by git tag `golden-lyamappo-v50-32k`.


## Directory layout

```
checkpoints/
├─ LyaMAPPO_lh4_32K/        # main paper model (proposed)
│   ├─ actor.pth            # Actor network weights (54→256→256→5)
│   ├─ actor_old.pth        # PPO old policy snapshot
│   ├─ critic.pth           # Critic weights (245→256→256→1)
│   ├─ actor_optimizer.pth  # Adam state (for training resume)
│   ├─ critic_optimizer.pth # Adam state
│   ├─ progress.pth         # update_count, best_completion_rate
│   └─ learning_curve.json  # 32 QuickEval points across training
│
└─ MAPPO_NoBat_lh4_8K/      # ablation: no battery in reward (8K only)
    └─ (same structure)
```

## How to load

```python
from core import Config, SatelliteMECEnv
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
cfg = _C()

env = SatelliteMECEnv(cfg)
mappo = MAPPOPolicy(cfg, name='MAPPO')
mappo.load('checkpoints/LyaMAPPO_lh4_32K')   # ← load trained weights

# eval
env.reset(phase='eval', seeds=cfg.get_eval_seeds(0))
mappo.set_eval_mode()
for _ in range(cfg.T_EVAL):
    _, _, _, info = env.step(policy=mappo)
print('CR:', env.get_eval_completion_rate())
```

## Training metadata

### LyaMAPPO_lh4_32K (proposed method)

| field | value |
|---|---|
| Training scenario | λ_high = 4.0 (high overload) |
| Training steps | 32 000 slots |
| Wall-clock training time | 25 min 24 s |
| Hyperparameters | BETA=0.02, W_DONE=10, W_HL=2, W_TIMEOUT=5, W_REJECT=5 |
| Final eval CR | 0.786 |
| Final eval HL/slot | 1.738e-4 |
| Final eval E2E delay | 3.335 s |
| Total update_count | 500 |
| Commit producing this | `c5d8f90` (LyaMAPPO methodology + final training) |

### MAPPO_NoBat_lh4_8K (ablation)

| field | value |
|---|---|
| Training scenario | λ_high = 4.0 |
| Training steps | 8 000 slots |
| Wall-clock training time | 5 min 44 s |
| Hyperparameters | identical to LyaMAPPO **except**: W_HL=0, Lyapunov battery terms disabled |
| Final eval CR | 0.822 (+3.6 pp vs LyaMAPPO due to no energy penalty) |
| Final eval HL/slot | 5.321e-4 (3.06× higher — confirms battery-reward causality) |
| Total update_count | 125 |
| Commit producing this | `8d91fcf` (ablation no-battery + plotting) |

### TD3Sched_lh4_16K (baseline, learned)

| field | value |
|---|---|
| Training scenario | λ_high = 4.0 |
| Training steps | 16 000 slots |
| Reward | no-battery Lyapunov cost (energy + queue drift; **no z_n / HL term**, per Huang TMC2024) |
| Final eval satisfaction | 0.837 |
| Final eval HL/slot | 4.43e-4 |

## ⚠️ Exact training commands (finalized hyperparameters)

**`config.py` defaults are NOT the finalized paper values** (defaults: `W_DONE=5,
W_HL=10, BETA=0.15, LAMBDA_HIGH=2.5`). The checkpoints were trained with the CLI
overrides below — these commands are the authoritative reproduction recipe.

```bash
# LyaMAPPO_lh4_32K (proposed)  — V=50, η=0.5, β_task=0.5 (config defaults), entropy β=0.02
python train_mappo_lambda4.py --t_train 32000 --v 50 --beta 0.02 \
    --w_done 10 --w_hl 2 --w_timeout 5 --w_reject 5 --n_runs 1 --tag lh4_32K

# MAPPO_NoBat_lh4_8K (ablation) — identical EXCEPT --no_battery
#   NOTE: --no_battery removes BOTH the Lyapunov battery/DoD action-cost AND sets
#   W_HL=0 (outcome HL reward). It is therefore a combined battery-ablation, not a
#   single-term ablation (disclosed; see WORK_LOG / paper ablation caveat).
python train_mappo_lambda4.py --t_train 8000 --v 50 --beta 0.02 \
    --w_done 10 --w_timeout 5 --w_reject 5 --no_battery --n_runs 1 --tag lh4_8K_nobat

# TD3Sched_lh4_16K (baseline)
python train_td3_sched.py --t_train 16000 --tag 16K
```

## Reproducing the paper headline result

```bash
# Re-evaluate from this checkpoint (no retraining)
python eval_pdfs.py --ckpt checkpoints/LyaMAPPO_lh4_32K

# Re-generate paper figures
python plot_paper.py \
    --full_dir results/<your-eval-result-dir> \
    --nobat_dir results/<ablation-result-dir>
```
