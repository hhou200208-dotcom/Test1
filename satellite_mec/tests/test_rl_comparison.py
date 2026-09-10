import unittest

from rl_comparison import ComparisonConfig
from rl_comparison.policies import ComparisonRewardMixin


class _Dummy(ComparisonRewardMixin):
    def __init__(self):
        self.cfg = ComparisonConfig()
        self.cfg.COMPARISON_HL_NORM = 2.0


class ComparisonRewardTests(unittest.TestCase):
    def test_common_reward_formula(self):
        rewards = _Dummy().build_comparison_rewards(
            per_sat_satisfied={n: int(n == 0) for n in range(25)},
            per_sat_timeout={n: 0 for n in range(25)},
            per_sat_rejected={n: 0 for n in range(25)},
            lifetime_losses=[2.0] + [0.0] * 24,
        )
        self.assertAlmostEqual(rewards[0], 0.0)
        self.assertTrue(all(rewards[n] == 0.0 for n in range(1, 25)))

    def test_physics_are_aligned(self):
        cfg = ComparisonConfig()
        self.assertEqual(cfg.N_SATS, 25)
        self.assertEqual(cfg.E_CAP, 54_000.0)
        self.assertEqual(cfg.DOD_MIN, 0.0)
        self.assertEqual(cfg.DOD_MAX, 0.8)
        self.assertEqual(cfg.KAPPA, 1.5e-27)


if __name__ == "__main__":
    unittest.main()
