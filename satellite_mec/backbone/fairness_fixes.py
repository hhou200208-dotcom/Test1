"""Fairness corrections discovered by the 1K backbone mini-pilot.

These adapters intentionally live separately from the original baselines so the main
project remains untouched while the course-project comparison can be audited.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import torch

from backbone.compare_algorithms import IPPOPolicy, SharedRewardMADDPG, system_reward


def actor_context_47(env, sat_id: int) -> np.ndarray:
    """Return the task-free information actually observable to one actor.

    Layout is exactly the first 47 dimensions of Satellite.get_state(): id(1), own(10),
    and four neighbour blocks (4*9).  It does NOT use the centralized critic vector.
    """
    cfg = env.cfg
    sat = env.constellation.satellites[int(sat_id)]
    # Use the environment's most recent exchange; fall back to a fresh exchange when
    # called at an external bootstrap point.
    global_info = getattr(env, "_global_info", None) or env.constellation.exchange_info()
    id_feat = [sat.sat_id / max(cfg.N_SATS - 1, 1)]
    local = [
        sat.qf_size / (cfg.Q_F_MAX + 1e-9),
        sat.qb_size / (cfg.Q_F_MAX + 1e-9),
        sat.nb_hat / max(cfg.MAX_DISPATCH, 1),
        sat.dod / cfg.DOD_MAX,
        sat.z_hat / max(cfg.Z_MAX, 1e-9),
        float(sat.xi),
        sat.tau_switch / cfg.ORBIT_PERIOD,
        sat.last_cpu_freq / max(cfg.CPU_FREQ, 1.0),
        sat.solar_power / max(cfg.P_SOLAR_MAX, 1e-6),
        max(cfg.DOD_MAX - sat.dod, 0.0) / cfg.DOD_MAX,
    ]
    nbr = []
    max_prop = cfg.ORBIT_RADIUS / cfg.SPEED_OF_LIGHT
    for neighbor_id in sat.neighbors:
        info = global_info.get(neighbor_id, {})
        nbr.extend([
            sat.link_rates.get(neighbor_id, cfg.B_AVG) / cfg.B_MAX,
            sat.prop_delays.get(neighbor_id, 0.0) / max(max_prop, 1e-9),
            (info.get("qf_size", 0.0) - cfg.THETA) / (cfg.Q_F_MAX + 1e-9),
            info.get("qb_size", 0.0) / (cfg.Q_F_MAX + 1e-9),
            info.get("nb", 0) / max(cfg.MAX_DISPATCH, 1),
            info.get("dod", 0.0) / cfg.DOD_MAX,
            float(info.get("xi", 1)),
            info.get("tau_switch", 0) / cfg.ORBIT_PERIOD,
            info.get("last_cpu_freq", 0.0) / max(cfg.CPU_FREQ, 1.0),
        ])
    x = np.asarray(id_feat + local + nbr, dtype=np.float32)
    expected = cfg.get_state_dim() - 7
    if x.shape != (expected,):
        raise RuntimeError(f"IPPO context dim mismatch: {x.shape} != {(expected,)}")
    return x


class FairIPPOPolicy(IPPOPolicy):
    """IPPO whose critic uses the actor's task-free local+neighbour observable context."""

    def collect_critic_values(self, env) -> None:
        self._slot_critic = {}
        for n in range(self.cfg.N_SATS):
            local = actor_context_47(env, n)
            x = torch.as_tensor(local, device=self.trainer.device)
            with torch.no_grad():
                value = float(self.critic.get_value(x).item())
            self._slot_critic[n] = (local, value)

    def _trigger_update(self, env):
        # MAPPOTrainer.compute_bootstrap is centralized, so provide the local bootstrap
        # explicitly while leaving the PPO update itself unchanged.
        if self.buffer.size() == 0:
            return None
        bootstrap = {}
        for n in range(self.cfg.N_SATS):
            local = actor_context_47(env, n)
            x = torch.as_tensor(local, device=self.trainer.device)
            with torch.no_grad():
                bootstrap[n] = float(self.critic.get_value(x).item())
        metrics = self.trainer.update(self.buffer, bootstrap)
        self.buffer.clear()
        if metrics:
            self.trainer.print_metrics(metrics)
        return metrics


class FairSharedRewardMADDPG(SharedRewardMADDPG):
    """MADDPG with slot-reward conservation and no false rollout terminals.

    The base learner stores one replay transition per task. Giving every task the full
    slot reward multiplies the return by the number of tasks and caused the 1K pilot's
    reward scale to collapse. Here the common R_sys is divided uniformly among executed
    tasks in the slot, so the sum of replay rewards for one physical slot equals R_sys.

    K_ROLLOUT boundaries are trajectory truncations, not environment terminals; pending
    transitions therefore continue across them. The final unmatched transition is simply
    discarded at the end of a finite training run instead of inventing a terminal state.
    """

    def _flush_slot(self, rewards: Dict, done: bool, info=None) -> None:
        if not self._slot_tasks:
            return
        per_task_reward = (system_reward(rewards) * self.reward_scale /
                           max(len(self._slot_tasks), 1))
        for (s, cobs, a, mask, _action_cost, _sat_id) in self._slot_tasks:
            if self._pending is not None:
                ps, pc, pa, pr = self._pending
                self.replay.add(ps, pc, pa, pr, s, cobs, mask, 0.0)
                self._env_steps += 1
            self._pending = (s, cobs, a, per_task_reward)
        # Deliberately ignore `done`: in SatelliteMECEnv it is a rollout boundary only.

    def finalize_training(self) -> None:
        # There is no genuine terminal state in the continuing simulator. Dropping one
        # unmatched final task transition is preferable to inserting a fake terminal.
        self._pending = None
