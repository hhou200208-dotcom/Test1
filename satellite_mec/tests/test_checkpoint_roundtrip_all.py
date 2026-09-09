"""Checkpoint save/load regression coverage for all four comparison algorithms."""
from __future__ import annotations

import pytest

from train_rl_reward_comparison import (
    ALGORITHMS,
    build_env,
    build_policy,
    make_config,
    new_history,
    save_checkpoint,
    verify_checkpoint_roundtrip,
)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_checkpoint_roundtrip_deterministic_outputs_all_algorithms(tmp_path, algorithm):
    cfg = make_config(10, 2)
    env = build_env(cfg, "train", {"task": 51, "task_param": 52, "dod_init": 53})
    policy = build_policy(algorithm, cfg, env)
    ckpt = tmp_path / algorithm.lower() / "checkpoint"
    history = new_history(algorithm, 42)
    save_checkpoint(ckpt, algorithm, policy, env, history, 0, 0.0)
    verify_checkpoint_roundtrip(algorithm, cfg, ckpt, policy)
