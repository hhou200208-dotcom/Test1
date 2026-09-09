"""Independent PPO baseline sharing the exact MAPPO actor architecture.

IPPO differs from MAPPO only in the critic information set: the actor receives
exactly the same 54-D per-task observation and uses ``training.networks.Actor``;
the critic receives the explicit 47-D local/no-task observation produced by the
environment's semantic node-state constructor.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .networks import RolloutBuffer
from .policy import MAPPOPolicy
from .trainer import MAPPOTrainer


LOCAL_CRITIC_DIM = 47


class _IPPOConfigView:
    """Forward every config field except the critic input dimension."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        # Pickle may probe attributes before ``_base`` has been restored.  Avoid
        # recursive access so resumable IPPO checkpoints can be reconstructed.
        base = self.__dict__.get("_base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def get_state_dim(self) -> int:
        return self._base.get_state_dim()

    def get_action_dim(self) -> int:
        return self._base.get_action_dim()

    def get_critic_state_dim(self) -> int:
        return LOCAL_CRITIC_DIM


class IPPOPolicy(MAPPOPolicy):
    """Parameter-sharing IPPO with a strictly local critic."""

    def __init__(self, config, name: str = "IPPO"):
        # Avoid constructing and then discarding a centralized trainer: initialize
        # the small common policy state directly.
        self.cfg = config
        self.name = name
        self.lyapunov_calc = None
        self._ippo_cfg = _IPPOConfigView(config)
        self.trainer = MAPPOTrainer(self._ippo_cfg)
        self.buffer = RolloutBuffer(self._ippo_cfg)
        self.actor = self.trainer.actor
        self.critic = self.trainer.critic
        self.learning_curve = []
        self._slot_inference = {}
        self._slot_critic = {}
        self._current_slot_t = 0
        self._eval_mode = False

    @staticmethod
    def _semantic_local_obs(env) -> Dict[int, np.ndarray]:
        if hasattr(env, "get_local_critic_obs"):
            obs = env.get_local_critic_obs()
        else:
            # ``SatelliteMECEnv.step`` invokes collect_critic_values with the base
            # environment, so use the same semantic constructor there rather than
            # relying on a fragile slice of the centralized observation.
            obs = {
                sat.sat_id: sat._get_node_state_47()
                for sat in env.constellation.satellites
            }
        out = {int(k): np.asarray(v, dtype=np.float32) for k, v in obs.items()}
        for n, value in out.items():
            if value.shape != (LOCAL_CRITIC_DIM,):
                raise ValueError(f"IPPO local critic obs for sat {n} has shape {value.shape}")
        return out

    def collect_critic_values(self, env) -> None:
        self._slot_critic = {}
        for n, critic_state in self._semantic_local_obs(env).items():
            state_t = torch.as_tensor(critic_state, dtype=torch.float32, device=self.trainer.device)
            with torch.no_grad():
                value = self.critic.get_value(state_t)
            self._slot_critic[n] = (critic_state.copy(), float(value.item()))

    def _trigger_update(self, env):
        if self.buffer.size() == 0:
            return None
        bootstrap = {}
        for n, critic_state in self._semantic_local_obs(env).items():
            state_t = torch.as_tensor(critic_state, dtype=torch.float32, device=self.trainer.device)
            with torch.no_grad():
                bootstrap[n] = float(self.critic.get_value(state_t).item())
        metrics = self.trainer.update(self.buffer, bootstrap)
        self.buffer.clear()
        if metrics:
            self.trainer.print_metrics(metrics)
        return metrics
