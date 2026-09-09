"""Unified reward used by the four-algorithm convergence experiment.

The legacy environment mixes Lyapunov action cost with phase-dependent outcome
shaping.  ``CommonRewardEnv`` deliberately runs the legacy environment with all
legacy outcome weights (and the completion bonus) disabled and then applies the
same pure ``common_reward_v1`` calculation in both training and evaluation.

This module is intentionally experiment-facing: existing paper checkpoints and
legacy reward paths are not changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


REWARD_DEFINITION = "common_reward_v1"
REWARD_COMPONENTS = (
    "action_cost", "done", "timeout", "reject", "health_loss", "queue"
)


@dataclass(frozen=True)
class CommonRewardSpec:
    """Frozen definition of ``common_reward_v1``."""

    lambda_high: float = 4.0
    w_done: float = 10.0
    w_timeout: float = 5.0
    w_reject: float = 5.0
    w_hl: float = 2.0
    w_queue: float = 0.05
    beta: float = 0.02
    beta_task: float = 0.5
    v: float = 50.0
    eta: float = 0.5

    def to_dict(self) -> Dict[str, float | str]:
        return {"reward_definition": REWARD_DEFINITION, **asdict(self)}


COMMON_REWARD_V1 = CommonRewardSpec()


def compute_common_reward_v1(
    action_cost_by_sat: Mapping[int, float],
    *,
    satisfied: int,
    timeout: int,
    rejected: int,
    total_health_loss: float,
    total_queue_bytes: float,
    hl_norm: float,
    queue_norm: float,
    spec: CommonRewardSpec = COMMON_REWARD_V1,
) -> Dict[str, float]:
    """Return one system-level slot reward ledger.

    ``action_cost_by_sat`` contains the raw rewards returned by
    ``Satellite.apply_action`` (negative normalized Lyapunov costs).  All other
    components are computed exactly once here.  Health loss and queue pressure
    are system sums normalized by the same per-satellite scales used by the
    environment, which keeps the system reward additive across satellites.
    """

    hl_den = max(float(hl_norm), 1e-12)
    q_den = max(float(queue_norm), 1.0)
    ledger = {
        "action_cost": float(fsum(float(v) for v in action_cost_by_sat.values())),
        "done": float(spec.w_done * int(satisfied)),
        "timeout": float(-spec.w_timeout * int(timeout)),
        "reject": float(-spec.w_reject * int(rejected)),
        "health_loss": float(-spec.w_hl * float(total_health_loss) / hl_den),
        "queue": float(-spec.w_queue * float(total_queue_bytes) / q_den),
    }
    ledger["total"] = float(fsum(ledger[k] for k in REWARD_COMPONENTS))
    return ledger


def distribute_system_reward(
    action_cost_by_sat: Mapping[int, float],
    ledger: Mapping[str, float],
    n_sats: int,
) -> Dict[int, float]:
    """Distribute non-action components without changing the system total.

    The environment has a per-agent reward API while the requested comparison
    uses one common system reward.  We retain each agent's own action cost and
    share the common outcome term uniformly.  Consequently ``sum(rewards)`` is
    exactly the ledger total (up to floating-point summation precision).
    """

    n = max(int(n_sats), 1)
    outcome = float(ledger["total"] - ledger["action_cost"])
    share = outcome / n
    rewards = {i: float(action_cost_by_sat.get(i, 0.0) + share) for i in range(n)}
    # Put any last-bit rounding residual on agent 0 so the invariant is exact.
    residual = float(ledger["total"] - fsum(rewards.values()))
    rewards[0] += residual
    return rewards


class CommonRewardEnv:
    """Thin wrapper applying ``common_reward_v1`` in train and eval identically.

    The wrapped base environment must have its legacy shaping disabled.  This
    class also derives the slot timeout count *after* the complete base step, so
    forward-queue, compute-queue, and same-slot transit timeouts are all counted.
    """

    def __init__(self, base_env, spec: CommonRewardSpec = COMMON_REWARD_V1):
        self.base_env = base_env
        self.spec = spec
        self._episode_timeout_after_last_step = int(base_env.episode_timeout)

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def reset(self, phase: str = "train", seeds: Optional[Dict[str, int]] = None):
        out = self.base_env.reset(phase=phase, seeds=seeds)
        self._episode_timeout_after_last_step = int(self.base_env.episode_timeout)
        return out

    def step(self, actions=None, policy=None):
        next_obs, raw_rewards, done, info = self.base_env.step(actions=actions, policy=policy)
        corrected_timeout = int(
            self.base_env.episode_timeout - self._episode_timeout_after_last_step
        )
        self._episode_timeout_after_last_step = int(self.base_env.episode_timeout)

        n_sats = int(self.cfg.N_SATS)
        total_hl = float(info.get("avg_health_loss", 0.0)) * n_sats
        total_queue = float(info.get("total_queue_size", 0.0)) * n_sats
        ledger = compute_common_reward_v1(
            raw_rewards,
            satisfied=int(info.get("slot_satisfied", 0)),
            timeout=corrected_timeout,
            rejected=int(info.get("rejected", 0)),
            total_health_loss=total_hl,
            total_queue_bytes=total_queue,
            hl_norm=float(self.cfg.HL_NORM),
            queue_norm=float(self.cfg.QUEUE_NORM),
            spec=self.spec,
        )
        rewards = distribute_system_reward(raw_rewards, ledger, n_sats)

        info = dict(info)
        info["slot_timeout"] = corrected_timeout
        info["reward_definition"] = REWARD_DEFINITION
        info["reward_ledger"] = ledger
        info["raw_action_cost_by_sat"] = {int(k): float(v) for k, v in raw_rewards.items()}
        return next_obs, rewards, done, info

    def get_local_critic_obs(self) -> Dict[int, np.ndarray]:
        """Explicit 47-D local/no-task critic observations for IPPO.

        This intentionally calls the semantic node-state constructor rather
        than slicing the first 47 coordinates of a 245-D centralized vector.
        """

        return {
            sat.sat_id: np.asarray(sat._get_node_state_47(), dtype=np.float32).copy()
            for sat in self.constellation.satellites
        }


def assert_legacy_shaping_disabled(cfg) -> None:
    """Fail fast if the base env could double-count outcome rewards."""

    fields = ("W_DONE", "W_TIMEOUT", "W_REJECT", "W_HL", "W_QUEUE", "W_DOD",
              "COMPLETION_BONUS")
    bad = {f: float(getattr(cfg, f, 0.0)) for f in fields if float(getattr(cfg, f, 0.0)) != 0.0}
    if bad:
        raise ValueError(f"legacy reward shaping must be disabled for {REWARD_DEFINITION}: {bad}")
