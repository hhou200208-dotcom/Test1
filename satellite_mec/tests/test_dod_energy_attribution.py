"""Tests for the satellite-slot energy ledger and offline attribution math."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from core import Config, SatelliteMECEnv
from plot_dod_energy_attribution import (
    ALGORITHMS,
    dod_bin,
    finalize_summary,
    read_and_aggregate,
)


class LedgerConfig(Config):
    T_TOTAL = 6_000


class TestSatelliteSlotLedger(unittest.TestCase):
    def test_environment_exports_complete_per_satellite_arrays(self):
        cfg = LedgerConfig()
        env = SatelliteMECEnv(cfg)
        env.reset(phase="eval", seeds={"task": 42, "task_param": 43, "dod_init": 44})
        _, _, _, info = env.step(actions={})
        keys = (
            "per_sat_battery_start_j", "per_sat_battery_end_j",
            "per_sat_battery_counterfactual_j", "per_sat_solar_energy_j",
            "per_sat_battery_counterfactual_physical_j",
            "per_sat_base_energy_j", "per_sat_compute_energy_j",
            "per_sat_tx_energy_j", "per_sat_arrived", "per_sat_completed",
            "per_sat_ontime", "per_sat_completed_bits", "per_sat_delay_sum",
        )
        for key in keys:
            self.assertEqual(len(info[key]), cfg.N_SATS, key)
        for n in range(cfg.N_SATS):
            expected = min(
                cfg.E_CAP,
                info["per_sat_battery_start_j"][n]
                + info["per_sat_solar_energy_j"][n]
                - info["per_sat_base_energy_j"][n],
            )
            self.assertAlmostEqual(info["per_sat_battery_counterfactual_j"][n], expected)
            expected_physical = max(cfg.E_CAP * (1.0 - cfg.DOD_MAX), expected)
            self.assertAlmostEqual(
                info["per_sat_battery_counterfactual_physical_j"][n], expected_physical
            )
            expected_end = min(
                cfg.E_CAP * (1.0 - cfg.DOD_MIN),
                max(
                    cfg.E_CAP * (1.0 - cfg.DOD_MAX),
                    info["per_sat_battery_start_j"][n]
                    + info["per_sat_solar_energy_j"][n]
                    - info["per_sat_base_energy_j"][n]
                    - info["per_sat_compute_energy_j"][n]
                    - info["per_sat_tx_energy_j"][n],
                ),
            )
            self.assertAlmostEqual(info["per_sat_battery_end_j"][n], expected_end)

    def test_fixed_dod_bins_include_upper_endpoint(self):
        self.assertEqual(dod_bin(0.0), 0)
        self.assertEqual(dod_bin(0.2), 1)
        self.assertEqual(dod_bin(0.6), 3)
        self.assertEqual(dod_bin(0.8), 3)

    def test_csv_is_the_only_metric_input(self):
        fields = (
            "algorithm", "seed", "scenario", "time_slot", "eval_slot", "satellite_id",
            "battery_start_j", "battery_capacity_j", "battery_end_j",
            "battery_counterfactual_j", "solar_energy_j", "base_energy_j",
            "battery_counterfactual_physical_j",
            "compute_energy_j", "tx_energy_j", "arrived_tasks", "completed_tasks",
            "ontime_tasks", "timeout_tasks", "rejected_tasks", "completed_bits",
            "completion_delay_sum_s",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for algorithm in ALGORITHMS:
                    writer.writerow({
                        "algorithm": algorithm, "seed": 42, "scenario": "test",
                        "time_slot": 1, "eval_slot": 0, "satellite_id": 0,
                        "battery_start_j": 43_200, "battery_capacity_j": 54_000,
                        "battery_end_j": 43_190, "battery_counterfactual_j": 43_200,
                        "battery_counterfactual_physical_j": 43_200,
                        "solar_energy_j": 5, "base_energy_j": 5,
                        "compute_energy_j": 8, "tx_energy_j": 2,
                        "arrived_tasks": 1, "completed_tasks": 1, "ontime_tasks": 1,
                        "timeout_tasks": 0, "rejected_tasks": 0,
                        "completed_bits": 10_000_000, "completion_delay_sum_s": 2,
                    })
            stats, _, meta = read_and_aggregate(path)
            summary = finalize_summary(stats, meta)
            for algorithm in ALGORITHMS:
                got = summary["algorithms"][algorithm]
                self.assertAlmostEqual(got["task_energy_per_completed_task_j"], 10.0)
                self.assertAlmostEqual(got["completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
