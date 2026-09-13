"""Tests for the frozen-policy constellation scalability experiment."""

import tempfile
import unittest
from pathlib import Path

from core import SatelliteMECEnv
from baselines import MHSPOPolicy
from run_scalability_evaluation import (
    ALGORITHMS,
    clone_warm_state,
    load_warmup_snapshot,
    make_config,
    save_warmup_snapshot,
    validate_topology,
)


class ScalabilityEvaluationTest(unittest.TestCase):
    def test_all_requested_topologies_have_four_symmetric_neighbors(self):
        for sats_per_plane in (8, 10, 12, 16, 20):
            cfg = make_config(16, sats_per_plane, 42, 2, 3)
            env = SatelliteMECEnv(cfg)
            validate_topology(env)
            self.assertEqual(cfg.N_SATS, 16 * sats_per_plane)
            self.assertEqual(env.constellation.neighbor_matrix.shape, (cfg.N_SATS, 4))

    def test_config_dimensions_are_scale_invariant(self):
        dimensions = set()
        for sats_per_plane in (8, 10, 12, 16, 20):
            cfg = make_config(16, sats_per_plane, 42, 2, 3)
            dimensions.add((
                cfg.get_state_dim(), cfg.get_critic_state_dim(), cfg.get_action_dim()
            ))
        self.assertEqual(dimensions, {(54, 245, 5)})

    def test_snapshot_round_trip_and_checksum_detection(self):
        cfg = make_config(16, 12, 42, 2, 3)
        env = SatelliteMECEnv(cfg)
        env.reset(phase="warmup", seeds={"task": 142, "task_param": 143, "dod_init": 144})
        env.current_slot = 2
        mhspo = MHSPOPolicy(cfg, env)
        checkpoint = Path(__file__).resolve().parents[1] / "checkpoints" / "LyaMAPPO_lh4_32K"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pkl.gz"
            metadata = save_warmup_snapshot(
                env, mhspo.predictors, path,
                seed=42, warmup_slots=2, checkpoint=checkpoint
            )
            restored, predictors, verified = load_warmup_snapshot(path)
            self.assertEqual(restored.current_slot, 2)
            self.assertEqual(restored.cfg.N_SATS, 192)
            self.assertEqual(len(predictors), 192)
            self.assertEqual(metadata["payload_sha256"], verified["payload_sha256"])

            clone, cloned_predictors = clone_warm_state(restored, predictors)
            self.assertIsNot(clone, restored)
            self.assertEqual(len(cloned_predictors), 192)

            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "compressed SHA-256"):
                load_warmup_snapshot(path)

    def test_requested_five_algorithms_are_explicit(self):
        self.assertEqual(
            ALGORITHMS,
            ("BLA-MAPPO", "MHSPO", "LyDRL-DoD", "GDCO", "LSO"),
        )


if __name__ == "__main__":
    unittest.main()
