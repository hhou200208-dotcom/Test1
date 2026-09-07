# Unified-reward 1K debug run (seed 42)

These files are the unedited outputs of the completed four-algorithm 1,000-slot
debug run. Each policy was evaluated deterministically for 500 slots at step 0
and step 1,000 using evaluation seed 1042. They are **not** a convergence study
and must not be presented as the requested 8K pilot or 32K preliminary result.

Raw checkpoints remain under the ignored directory
`results/rl_reward_comparison/debug_seed_42/`. Continue with fresh complete runs:

```bash
python train_rl_reward_comparison.py --steps 8000 --eval-interval 2000 --eval-slots 500
python plot_rl_reward_comparison.py --input results/rl_reward_comparison/seed_42 \
  --publish docs/rl_reward_comparison/seed_42
```
