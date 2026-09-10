"""Dependency-light tests for the new BLA-MAPPO mathematical core."""

import unittest

import numpy as np

from core import Config, SatelliteMECEnv
from new_bla_mappo.config import NewBLAMAPPOConfig
from new_bla_mappo.buffer import PaperRolloutBuffer
from new_bla_mappo.lifetime import (
    EmpiricalMeanNormalizer,
    induced_lifetime_loss,
    lifetime_curve,
    net_discharge_loss,
)
from train_new_bla_mappo import analyze_plateau


class _Config:
    GAMMA = 0.99
    LAMBDA_GAE = 0.95
    BLA_ETA_C = 0.1


class TestLifetimeEquations(unittest.TestCase):
    def test_curve_is_increasing_and_convex(self):
        values = [lifetime_curve(x) for x in (0.2, 0.4, 0.6, 0.8)]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))
        increments = np.diff(values)
        self.assertTrue(all(b > a for a, b in zip(increments, increments[1:])))

    def test_net_discharge_clips_charging_to_zero(self):
        self.assertEqual(net_discharge_loss(0.5, 0.4), 0.0)

    def test_induced_loss_is_zero_without_task_energy(self):
        self.assertEqual(induced_lifetime_loss(0.4, 30_000, 30_000, 54_000), 0.0)

    def test_same_energy_costs_more_at_high_dod(self):
        low = induced_lifetime_loss(0.2, 42_000, 43_000, 54_000)
        high = induced_lifetime_loss(0.7, 15_000, 16_000, 54_000)
        self.assertGreater(high, low)


class TestPaperConfiguration(unittest.TestCase):
    def test_paper_dimensions_and_scale(self):
        cfg = NewBLAMAPPOConfig()
        self.assertEqual(cfg.N_SATS, 192)
        self.assertEqual(cfg.get_state_dim(), 40)
        self.assertEqual(cfg.get_critic_state_dim(), 44)
        self.assertAlmostEqual(cfg.V_DVFS, 2.0e17)
        self.assertAlmostEqual(cfg.BLA_B_MIN, 0.2 * cfg.E_CAP)


class TestNormalizer(unittest.TestCase):
    def test_freezes_after_requested_slots(self):
        norm = EmpiricalMeanNormalizer(warmup_slots=2)
        norm.observe_slot([1.0, 3.0])
        norm.observe_slot([5.0, 7.0])
        self.assertTrue(norm.frozen)
        self.assertAlmostEqual(norm.value, 4.0)
        norm.observe_slot([100.0])
        self.assertAlmostEqual(norm.value, 4.0)


class TestPlateauAnalysis(unittest.TestCase):
    def test_flat_tail_is_plateau(self):
        result = analyze_plateau(np.ones(40, dtype=np.float64) * 2.0)
        self.assertTrue(result["plateau_detected"])

    def test_clear_rising_tail_is_not_plateau(self):
        result = analyze_plateau(np.linspace(0.0, 4.0, 40, dtype=np.float64))
        self.assertFalse(result["plateau_detected"])


class TestDualGranularityCredit(unittest.TestCase):
    def test_centered_cost_is_zero_sum_within_satellite_slot(self):
        buf = PaperRolloutBuffer(_Config())
        state = np.zeros(2, np.float32)
        mask = np.ones(2, np.float32)
        buf.add_slot(0, 0, state, system_reward=1.0, value=0.0)
        buf.add_task(0, 0, state, 0, 0.0, mask, structured_cost=-1.0)
        buf.add_task(0, 0, state, 1, 0.0, mask, structured_cost=1.0)
        buf.compute_advantages({0: 0.0})
        self.assertAlmostEqual(sum(t.centered_cost for t in buf.tasks), 0.0)
        self.assertLessEqual(max(abs(t.centered_cost) for t in buf.tasks), 1.0)


class _HookPolicy:
    """Tiny policy exercising the optional paper hooks without PyTorch."""

    def begin_slot(self, env):
        self.records = []

    def collect_critic_values(self, env):
        pass

    def build_action_mask(self, env, sat, task, t, neighbor_info, neighbor_nb):
        mask = np.zeros(env.cfg.get_action_dim(), np.float32)
        mask[0] = 1.0
        return mask

    def build_actor_state(self, env, sat, task, t, neighbor_info):
        return sat.get_state(task, t, neighbor_info)

    def act_one(self, state, mask):
        return 0, 0.0

    def task_structured_cost(self, env, sat, task, action, t, neighbor_info, commit):
        return 0.25

    def record_paper_task_transition(self, **kwargs):
        self.records.append(kwargs)

    def compute_slot_lifetime_losses(self, env, dod_before_map):
        return [0.0] * env.cfg.N_SATS


class TestEnvironmentHooks(unittest.TestCase):
    def test_paper_policy_receives_decomposed_info(self):
        cfg = Config()
        env = SatelliteMECEnv(cfg)
        env.reset(phase="train", seeds={"task": 1, "task_param": 2, "dod_init": 3})
        policy = _HookPolicy()
        _, _, _, info = env.step(policy=policy)
        self.assertEqual(len(info["per_sat_satisfied"]), cfg.N_SATS)
        self.assertEqual(len(info["per_sat_timeout"]), cfg.N_SATS)
        self.assertEqual(len(info["per_sat_rejected"]), cfg.N_SATS)
        self.assertEqual(len(info["paper_lifetime_losses"]), cfg.N_SATS)


if __name__ == "__main__":
    unittest.main()
