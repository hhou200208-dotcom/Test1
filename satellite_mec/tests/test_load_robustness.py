"""Tests for the frozen BLA-MAPPO business-load robustness experiment."""

import csv
import tempfile
import unittest
from pathlib import Path

from run_load_robustness import (
    BASE_LAMBDA_HIGH,
    BASE_LAMBDA_LOW,
    DEFAULT_ALPHAS,
    SUMMARY_FIELDS,
    make_config,
    plot_results,
)


class LoadRobustnessTest(unittest.TestCase):
    def test_all_rates_are_uniformly_scaled_on_native_5x5_topology(self):
        high_sets = []
        derived_constants = []
        for alpha in DEFAULT_ALPHAS:
            cfg = make_config(alpha, 42, 2, 3)
            self.assertEqual((cfg.N_PLANES, cfg.N_SATS_PER_PLANE, cfg.N_SATS), (5, 5, 25))
            self.assertAlmostEqual(cfg.LAMBDA_HIGH, BASE_LAMBDA_HIGH * alpha)
            self.assertAlmostEqual(cfg.LAMBDA_LOW, BASE_LAMBDA_LOW * alpha)
            self.assertEqual(cfg.T_TOTAL, 2 + 3 + cfg.ORBIT_PERIOD)
            # The fixed load-hotspot set depends only on SEED_LINK and N_SATS.
            from core import SatelliteMECEnv
            high_sets.append(frozenset(SatelliteMECEnv(cfg).constellation.high_load_sats))
            derived_constants.append((cfg.LAMBDA_MAX, cfg.THETA_NUM, cfg.Q_NORM))
        self.assertEqual(len(set(high_sets)), 1)
        self.assertEqual(len(set(derived_constants)), 1)

    def test_baseline_alpha_exactly_matches_gold_training_load(self):
        cfg = make_config(1.0, 42, 2, 3)
        self.assertEqual(cfg.LAMBDA_HIGH, 4.0)
        self.assertEqual(cfg.LAMBDA_LOW, 0.1)
        self.assertEqual(cfg.V_DVFS, 2e17)

    def test_plotter_creates_four_curves_and_overview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                for alpha in DEFAULT_ALPHAS:
                    row = {field: 0 for field in SUMMARY_FIELDS}
                    row.update({
                        "algorithm": "BLA-MAPPO", "alpha": alpha,
                        "user_satisfaction": 0.8,
                        "mean_system_delay_per_slot_s": alpha,
                        "mean_system_energy_per_slot_j": alpha,
                        "total_lifetime_loss": alpha,
                    })
                    writer.writerow(row)
            paths = plot_results(summary, root / "figures")
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
