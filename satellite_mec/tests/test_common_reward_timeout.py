"""Timeout accounting regression: include timeouts that occur late in a slot."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.common_reward import CommonRewardEnv


class _FakeBaseEnv:
    def __init__(self):
        self.episode_timeout = 10
        self.cfg = SimpleNamespace(N_SATS=2, HL_NORM=1.0, QUEUE_NORM=100.0)

    def step(self, actions=None, policy=None):
        # Simulate 2 queue timeouts already visible in the legacy per-slot info,
        # followed by one transit timeout later in the same base-env step.
        self.episode_timeout += 3
        info = {
            "slot_timeout": 2,
            "slot_satisfied": 0,
            "rejected": 0,
            "avg_health_loss": 0.0,
            "total_queue_size": 0.0,
        }
        return {}, {0: -1.0, 1: -2.0}, False, info


def test_post_step_episode_timeout_delta_includes_late_transit_timeout():
    env = CommonRewardEnv(_FakeBaseEnv())
    _, rewards, _, info = env.step(actions={})
    assert info["slot_timeout"] == 3
    assert info["reward_ledger"]["timeout"] == -15.0
    assert info["reward_ledger"]["action_cost"] == -3.0
    assert sum(rewards.values()) == pytest.approx(info["reward_ledger"]["total"])
