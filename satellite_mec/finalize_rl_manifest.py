#!/usr/bin/env python3
"""Finalize reproducibility metadata after plots have been generated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_rl_reward_comparison import make_config


def finalize(root: Path, steps: int, eval_slots: int) -> None:
    path = root / "experiment_manifest.json"
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    cfg = make_config(steps, eval_slots)
    manifest["algorithm_hyperparameters"] = {
        "MAPPO": {
            "actor_lr": cfg.LR_ACTOR,
            "critic_lr": cfg.LR_CRITIC,
            "gamma": cfg.GAMMA,
            "gae_lambda": cfg.LAMBDA_GAE,
            "ppo_clip": cfg.EPS_CLIP,
            "epochs_per_rollout": cfg.EPOCH,
            "minibatch": cfg.MINIBATCH,
            "entropy_beta": cfg.BETA,
            "task_advantage_beta": cfg.BETA_TASK,
            "rollout_slots": cfg.K_ROLLOUT,
        },
        "IPPO": {
            "actor_lr": cfg.LR_ACTOR,
            "critic_lr": cfg.LR_CRITIC,
            "gamma": cfg.GAMMA,
            "gae_lambda": cfg.LAMBDA_GAE,
            "ppo_clip": cfg.EPS_CLIP,
            "epochs_per_rollout": cfg.EPOCH,
            "minibatch": cfg.MINIBATCH,
            "entropy_beta": cfg.BETA,
            "task_advantage_beta": cfg.BETA_TASK,
            "rollout_slots": cfg.K_ROLLOUT,
            "critic_information": "explicit local 47-D node observation only",
        },
        "MADDPG": {
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "tau": 0.01,
            "batch_size": 256,
            "replay_capacity": 1_000_000,
            "start_steps": 2000,
            "updates_per_slot": 1,
            "expl_noise": 0.2,
            "gumbel_temp_start": 1.0,
            "gumbel_temp_end": 0.5,
            "gumbel_decay_slots": 30000,
            "zhong_reward": False,
            "reward_scale": 1.0,
        },
        "TD3": {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "replay_capacity": 1_000_000,
            "start_steps": 2000,
            "updates_per_slot": 1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "expl_noise": 0.1,
            "reward_scale": 1.0,
        },
    }
    manifest["checkpoint_type"] = "full training/resume checkpoint; not inference-only"
    manifest["raw_results_ignored_by_git"] = True
    manifest["output_files"] = sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eval-slots", type=int, default=500)
    args = p.parse_args(argv)
    finalize(args.root, args.steps, args.eval_slots)
    print(f"finalized {args.root / 'experiment_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
