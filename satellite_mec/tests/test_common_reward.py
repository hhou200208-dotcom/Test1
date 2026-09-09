"""Regression tests for the unified four-algorithm comparison protocol."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.common_reward import (
    COMMON_REWARD_V1,
    REWARD_COMPONENTS,
    REWARD_DEFINITION,
    compute_common_reward_v1,
)
from core.task import Task
from train_rl_reward_comparison import (
    ALGORITHMS,
    CountedIPPO,
    CountedMADDPG,
    CountedMAPPO,
    CountedTD3,
    append_history,
    build_env,
    build_policy,
    evaluate_fixed,
    make_config,
    new_history,
    policy_fingerprint,
    save_checkpoint,
    verify_checkpoint_roundtrip,
)
from training.ippo import LOCAL_CRITIC_DIM


def test_common_reward_configuration_is_exact():
    s = COMMON_REWARD_V1
    assert s.lambda_high == 4.0
    assert s.w_done == 10.0
    assert s.w_timeout == 5.0
    assert s.w_reject == 5.0
    assert s.w_hl == 2.0
    assert s.w_queue == 0.05
    assert s.beta == 0.02
    assert s.beta_task == 0.5
    assert s.v == 50.0
    assert s.eta == 0.5


def test_reward_ledger_total_exact_component_sum():
    ledger = compute_common_reward_v1(
        {0: -1.25, 1: -2.5}, satisfied=3, timeout=2, rejected=1,
        total_health_loss=2e-4, total_queue_bytes=5e8,
        hl_norm=1e-4, queue_norm=1e9,
    )
    assert ledger["total"] == pytest.approx(sum(ledger[k] for k in REWARD_COMPONENTS), abs=1e-12)
    assert ledger["action_cost"] == pytest.approx(-3.75)
    assert ledger["done"] == 30.0
    assert ledger["timeout"] == -10.0
    assert ledger["reject"] == -5.0


def test_train_and_eval_use_identical_common_reward():
    cfg1 = make_config(10, 3)
    cfg2 = make_config(10, 3)
    seeds = {"task": 777, "task_param": 778, "dod_init": 779}
    train = build_env(cfg1, "train", seeds)
    evalu = build_env(cfg2, "eval", seeds)
    for _ in range(4):
        _, r1, _, i1 = train.step(actions={})
        _, r2, _, i2 = evalu.step(actions={})
        assert i1["reward_definition"] == REWARD_DEFINITION
        assert i2["reward_definition"] == REWARD_DEFINITION
        assert i1["reward_ledger"] == pytest.approx(i2["reward_ledger"])
        assert sum(r1.values()) == pytest.approx(i1["reward_ledger"]["total"])
        assert sum(r2.values()) == pytest.approx(i2["reward_ledger"]["total"])


def test_same_fixed_action_trajectory_has_identical_ledger_for_four_algorithm_labels():
    ledgers = {}
    seeds = {"task": 1234, "task_param": 1235, "dod_init": 1236}
    for alg in ALGORITHMS:
        env = build_env(make_config(10, 2), "train", seeds)
        seq = []
        for _ in range(5):
            # Empty action lists invoke the environment's deterministic first-legal
            # fallback; therefore all four labelled runs execute the same legal path.
            _, _, _, info = env.step(actions={})
            seq.append(tuple(info["reward_ledger"][k] for k in (*REWARD_COMPONENTS, "total")))
        ledgers[alg] = np.asarray(seq)
    for alg in ALGORITHMS[1:]:
        np.testing.assert_array_equal(ledgers[alg], ledgers["MAPPO"])


def test_completion_bonus_cannot_leak_into_action_cost():
    cfg = make_config(10, 2)
    assert cfg.COMPLETION_BONUS == 0.0
    ledger = compute_common_reward_v1(
        {0: -7.0}, satisfied=2, timeout=0, rejected=0,
        total_health_loss=0.0, total_queue_bytes=0.0,
        hl_norm=cfg.HL_NORM, queue_norm=cfg.QUEUE_NORM,
    )
    assert ledger["action_cost"] == -7.0
    assert ledger["done"] == 20.0


def test_actor_and_critic_dimensions_and_mappo_ippo_actor_identity():
    cfg = make_config(10, 2)
    m = CountedMAPPO(cfg)
    i = CountedIPPO(cfg)
    assert cfg.get_state_dim() == 54
    assert cfg.get_critic_state_dim() == 245
    assert LOCAL_CRITIC_DIM == 47
    assert type(m.actor) is type(i.actor)
    assert list(m.actor.state_dict()) == list(i.actor.state_dict())
    assert sum(p.numel() for p in m.actor.parameters()) == sum(p.numel() for p in i.actor.parameters())
    assert m.critic.fc1.in_features == 245
    assert i.critic.fc1.in_features == 47


def test_explicit_local_critic_obs_is_semantic_not_central_slice():
    cfg = make_config(10, 2)
    env = build_env(cfg, "train", {"task": 1, "task_param": 2, "dod_init": 3})
    local = env.get_local_critic_obs()
    central = env.get_critic_obs()
    assert all(v.shape == (47,) for v in local.values())
    assert all(v.shape == (245,) for v in central.values())
    # Explicit semantic source must exactly match the satellite node-state builder.
    for sat in env.constellation.satellites:
        np.testing.assert_array_equal(local[sat.sat_id], sat._get_node_state_47())


def _find_state_and_mask(env):
    # Generate until at least one task exists, then use the environment's own mask.
    for _ in range(20):
        env._global_info = env.constellation.exchange_info()
        tasks = env.constellation.generate_tasks(env.current_slot)
        for sat in env.constellation.satellites:
            incoming = tasks.get(sat.sat_id, [])
            if incoming:
                task = incoming[0]
                sat.admit_task(task); sat.init_temp_state()
                nbr_nb = {nid: 0 for nid in sat.neighbors}
                mask = sat.get_action_mask(task, env.current_slot, nbr_nb)
                state = sat.get_state(task, env.current_slot,
                                      {nid: env._global_info[nid] for nid in sat.neighbors})
                if mask.sum() > 0:
                    return state, mask
    raise AssertionError("failed to generate a legal task")


@pytest.mark.parametrize("alg", ["MADDPG", "TD3"])
def test_offpolicy_final_action_respects_same_five_way_mask_and_replay_audit(alg):
    cfg = make_config(10, 2)
    env = build_env(cfg, "train", {"task": 11, "task_param": 12, "dod_init": 13})
    p = build_policy(alg, cfg, env)
    state = np.zeros(cfg.get_state_dim(), dtype=np.float32)
    mask = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    action, _ = p.act_one(state, mask)
    assert 0 <= action < 5
    assert mask[action] == 1
    p.collect_critic_values(env.base_env) if alg == "MADDPG" else None
    p.record_task_transition(0, 0, state, action, 0.0, mask, -1.0)
    # Force terminal flush so the first transition is materialized.
    if alg == "MADDPG":
        p._flush_slot({0: -1.0}, True, {})
    else:
        p._flush_slot({0: -1.0}, True)
    assert p.replay.size == 1
    assert int(p.replay.executed_action[0]) == action
    np.testing.assert_array_equal(p.replay.executed_mask[0], mask)
    assert p.replay.network_action[0].shape == (5,)


@pytest.mark.parametrize("alg", ALGORITHMS)
def test_fixed_evaluation_does_not_mutate_training_state(alg):
    cfg = make_config(10, 3)
    env = build_env(cfg, "train", {"task": 21, "task_param": 22, "dod_init": 23})
    p = build_policy(alg, cfg, env)
    before = policy_fingerprint(alg, p)
    metrics = evaluate_fixed(alg, p, cfg, slots=3)
    after = policy_fingerprint(alg, p)
    assert before == after
    assert math.isfinite(metrics["eval_return_per_slot"])


def test_checkpoint_roundtrip_preserves_deterministic_actor_and_full_ppo_state(tmp_path):
    cfg = make_config(10, 2)
    env = build_env(cfg, "train", {"task": 31, "task_param": 32, "dod_init": 33})
    p = CountedMAPPO(cfg)
    ckpt = tmp_path / "checkpoint"
    h = new_history("MAPPO", 42)
    save_checkpoint(ckpt, "MAPPO", p, env, h, 0, 0.0)
    verify_checkpoint_roundtrip("MAPPO", cfg, ckpt, p)


def test_history_schema_lengths_eval_points_and_nan_guard():
    h = new_history("MAPPO", 42)
    class P:
        gradient_updates = 0
        actor_gradient_updates = 0
        critic_gradient_updates = 0
    metrics = {
        "eval_return_total": 1.0, "eval_return_per_slot": 0.002,
        "completion_rate": 0.5, "satisfaction": 0.4, "avg_delay": 1.0,
        "avg_health_loss": 1e-5, "queue_tasks_per_sat": 2.0,
        "reward_ledger": {k: 0.0 for k in (*REWARD_COMPONENTS, "total")},
    }
    for step in range(0, 32001, 2000):
        append_history(h, step, float(step), 0.0, P(), metrics)
    assert h["environment_steps"] == list(range(0, 32001, 2000))
    assert len(h["environment_steps"]) == 17
    for values in h["reward_ledger"].values():
        assert len(values) == 17
    bad = dict(metrics)
    bad["avg_delay"] = float("nan")
    with pytest.raises(AssertionError):
        append_history(new_history("MAPPO", 42), 0, 0.0, 0.0, P(), bad)


def test_td3_gradient_update_counter_is_real_optimizer_count():
    cfg = make_config(10, 2)
    env = build_env(cfg, "train", {"task": 41, "task_param": 42, "dod_init": 43})
    p = CountedTD3(cfg, env.base_env, seed=42, batch_size=1, start_steps=1,
                   reward_scale=1.0, policy_freq=2)
    z = np.zeros(cfg.get_state_dim(), np.float32)
    a = np.zeros(cfg.get_action_dim(), np.float32)
    p.replay.add(z, a, 0.0, z, 0.0)
    p._update()
    assert p.critic_gradient_updates == 1
    assert p.actor_gradient_updates == 0
    p._update()
    assert p.critic_gradient_updates == 2
    assert p.actor_gradient_updates == 1
    assert p.gradient_updates == 3
