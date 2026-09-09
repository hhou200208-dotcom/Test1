"""Fast regression tests for the current 54/245-dimensional implementation.

These tests intentionally track the current public API instead of retaining the
pre-54-D/245-D assertions from an older repository revision.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from baselines.deterministic import LocalOnlyPolicy
from core.config import Config
from core.constellation import Constellation
from core.env import SatelliteMECEnv
from core.lyapunov import LyapunovCalculator
from core.satellite import Satellite
from core.task import Task


def make_task(arrive=0, deadline=10.0, size=20e6, cpu=20.0):
    return Task(task_id=(arrive, 0, 0), size=size, cpu_cycles=cpu,
                deadline=deadline, arrive_slot=arrive, access_sat=0)


def test_config_current_dimensions_and_seeds():
    cfg = Config()
    assert cfg.get_state_dim() == 54
    assert cfg.get_critic_state_dim() == 245
    assert cfg.get_action_dim() == 5
    assert cfg.GLOBAL_SUMMARY_DIM == 10
    assert cfg.get_eval_seeds(0) != cfg.get_eval_seeds(1)
    assert set(cfg.get_eval_seeds(0)) == {"task", "task_param", "dod_init"}
    assert cfg.LAMBDA_MAX > 0
    assert cfg.Q_NORM > 0


def test_task_state_machine_and_timing():
    t = make_task(arrive=10, deadline=8.0)
    assert t.status == "queuing"
    assert t.remain_time(10) == pytest.approx(8.0)
    assert t.remain_time(15) == pytest.approx(3.0)
    assert not t.is_timeout(17)
    assert t.is_timeout(18)
    before = t.trans_delay_acc
    t.forward(200e6, 0.005)
    assert t.hops == 1
    assert t.trans_delay_acc > before
    t.update_processed(t.size)
    assert t.is_done()


def test_lyapunov_current_dvfs_aware_api_is_finite():
    cfg = Config()
    calc = LyapunovCalculator(cfg)
    task = make_task(deadline=12.0, size=10e6, cpu=10.0)
    sat_state = {
        "dod": 0.3, "n_f": 2, "qf_size": 2e7, "nb_hat": 1, "z_hat": 0.0,
        "q_cycles_hat": 2e8, "f_floor_hat": 1e8,
    }
    neighbor = {"qf_size": 1e7, "n_f": 1}
    assert calc.health_loss(0.5) > 0
    assert calc.delta_dod_comp(task, sat_state) >= 0
    assert calc.delta_dod_trans(task, 200e6) > 0
    assert math.isfinite(calc.normalized_local_cost(task, sat_state, 2, 0))
    assert math.isfinite(calc.normalized_forward_cost(task, sat_state, neighbor, 200e6, 0))
    assert calc.delta_dod_trans(task, 0.0) == 0.0


def make_satellite(cfg: Config) -> Satellite:
    return Satellite(
        sat_id=0,
        neighbors=[1, 2, 3, 4],
        link_rates={1: 200e6, 2: 180e6, 3: 160e6, 4: 220e6},
        prop_delays={1: 0.002, 2: 0.002, 3: 0.002, 4: 0.002},
        solar_seq=np.full(cfg.T_TOTAL, 15.0, dtype=np.float32),
        config=cfg,
    )


def test_satellite_current_actor_and_critic_shapes_and_masks():
    cfg = Config()
    sat = make_satellite(cfg)
    sat.init_temp_state()
    task = make_task(deadline=12.0, size=10e6, cpu=10.0)
    state = sat.get_state(task, 0, {})
    critic = sat.get_critic_state({}, 0, np.zeros(cfg.GLOBAL_SUMMARY_DIM, np.float32))
    local = sat._get_node_state_47()
    mask = sat.get_action_mask(task, 0, {n: 0 for n in sat.neighbors})
    assert state.shape == (54,)
    assert critic.shape == (245,)
    assert local.shape == (47,)
    assert mask.shape == (5,)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() >= 1
    sat.nb_hat = cfg.MAX_DISPATCH
    assert sat.get_action_mask(task, 0, {})[0] == 0.0


def test_satellite_queue_reset_and_timeout():
    cfg = Config()
    sat = make_satellite(cfg)
    task = make_task(deadline=2.0)
    assert sat.admit_task(task)
    assert len(sat.forward_queue) == 1
    removed = sat.remove_timeout_tasks(3)
    assert removed and removed[0].status == "timeout"
    sat.admit_task(make_task(deadline=10.0))
    sat.reset()
    assert not sat.forward_queue
    assert not sat.compute_queue
    assert sat.nb == 0


def test_constellation_topology_rng_and_stats_are_deterministic():
    cfg = Config()
    c1 = Constellation(cfg)
    c2 = Constellation(cfg)
    assert len(c1.satellites) == 25
    assert all(len(s.neighbors) == 4 for s in c1.satellites)
    np.testing.assert_array_equal(c1.neighbor_matrix, c2.neighbor_matrix)
    np.testing.assert_allclose(c1.link_rate_matrix, c2.link_rate_matrix)
    seeds = {"task": 901, "task_param": 902, "dod_init": 903}
    c1.reset(seeds); c2.reset(seeds)
    a = c1.generate_tasks(0); b = c2.generate_tasks(0)
    assert [len(a[n]) for n in range(25)] == [len(b[n]) for n in range(25)]
    stats = c1.get_stats()
    assert "avg_queue_tasks" in stats
    assert "avg_health_loss" in stats


def test_environment_step_returns_finite_rewards_and_ledger():
    cfg = Config()
    env = SatelliteMECEnv(cfg)
    env.reset("train", {"task": 1001, "task_param": 1002, "dod_init": 1003})
    _, rewards, done, info = env.step(actions={})
    assert len(rewards) == 25
    assert all(math.isfinite(v) for v in rewards.values())
    assert isinstance(done, bool)
    assert "reward_ledger" in info
    assert "queue_task_count" in info


def test_local_only_policy_actions_are_valid_type():
    cfg = Config()
    env = SatelliteMECEnv(cfg)
    env.reset("train", {"task": 1101, "task_param": 1102, "dod_init": 1103})
    policy = LocalOnlyPolicy(cfg, env)
    # One real environment step exercises the public PolicyInterface path.
    _, rewards, _, _ = env.step(policy=policy)
    assert all(math.isfinite(v) for v in rewards.values())
