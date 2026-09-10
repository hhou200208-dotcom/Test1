"""MAPPO, IPPO, MADDPG and QMIX under one reward/physics protocol."""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.maddpg_dod import MADDPGDoDPolicy
from core.config import Config
from new_bla_mappo.lifetime import induced_lifetime_loss
from training.policy import MAPPOPolicy


class ComparisonConfig(Config):
    """25-satellite configuration with the new paper battery parameters."""

    N_PLANES = 5
    N_SATS_PER_PLANE = 5
    N_SATS = 25
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0
    DOD_MAX = 0.8
    BETA = 0.02
    W_DONE = 10.0
    W_TIMEOUT = 5.0
    W_REJECT = 5.0
    COMPARISON_RHO_L = 10.0
    COMPARISON_HL_NORM = 3.0e-6

    def __init__(self):
        super().__init__()
        self.V_DVFS = 2.0e17


class IPPOConfig(ComparisonConfig):
    """IPPO uses a local critic with the same dimension as its actor state."""

    def get_critic_state_dim(self) -> int:
        return self.get_state_dim()


class ComparisonRewardMixin:
    """Exact Eq. (12) HL and one common per-satellite reward definition."""

    def compute_slot_lifetime_losses(self, env, dod_before_map) -> List[float]:
        cfg = self.cfg
        losses = []
        for sat in env.constellation.satellites:
            d0 = float(dod_before_map[sat.sat_id])
            battery = cfg.E_CAP * (1.0 - d0)
            base_energy = cfg.P_HOUSEKEEPING * cfg.TAU
            solar = sat.solar_power * cfg.TAU
            task_energy = sat.slot_comp_energy + sat.slot_trans_energy
            battery_base = min(cfg.E_CAP, battery + solar - base_energy)
            battery_actual = min(cfg.E_CAP, battery_base - task_energy)
            losses.append(induced_lifetime_loss(
                d0, battery_actual, battery_base, cfg.E_CAP, cfg.A_COEF))
        return losses

    def build_comparison_rewards(
        self, *, per_sat_satisfied, per_sat_timeout, per_sat_rejected,
        lifetime_losses,
    ) -> Dict[int, float]:
        cfg = self.cfg
        losses = lifetime_losses or [0.0] * cfg.N_SATS
        scale = max(float(cfg.COMPARISON_HL_NORM), 1e-12)
        return {
            n: float(
                cfg.W_DONE * per_sat_satisfied[n]
                - cfg.W_TIMEOUT * per_sat_timeout[n]
                - cfg.W_REJECT * per_sat_rejected[n]
                - cfg.COMPARISON_RHO_L * losses[n] / scale
            )
            for n in range(cfg.N_SATS)
        }

    @staticmethod
    def _attach_reward(rewards: Dict[int, float], info: dict) -> None:
        info["comparison_system_reward"] = float(np.mean(list(rewards.values())))


class ComparisonMAPPOPolicy(ComparisonRewardMixin, MAPPOPolicy):
    def __init__(self, config, name: str = "MAPPO"):
        super().__init__(config, name=name)

    def run_step(self, env):
        rewards, done, info = super().run_step(env)
        self._attach_reward(rewards, info)
        return rewards, done, info


class ComparisonIPPOPolicy(ComparisonRewardMixin, MAPPOPolicy):
    """Independent PPO: shared local actor and local value critic, no CTDE state."""

    def __init__(self, config, name: str = "IPPO"):
        super().__init__(config, name=name)

    def _local_critic_states(self, env) -> Dict[int, np.ndarray]:
        dim = self.cfg.get_critic_state_dim()
        return {
            # The first 47 entries are the satellite's own node block;
            # task-free padding keeps the critic strictly local.
            n: np.concatenate([
                np.asarray(s[:47], dtype=np.float32),
                np.zeros(max(dim - 47, 0), dtype=np.float32),
            ])[:dim]
            for n, s in env.get_critic_obs().items()
        }

    def collect_critic_values(self, env) -> None:
        self._slot_critic = {}
        for n, state in self._local_critic_states(env).items():
            state_t = torch.as_tensor(state, dtype=torch.float32,
                                      device=self.trainer.device)
            with torch.no_grad():
                value = self.critic.get_value(state_t)
            self._slot_critic[n] = (state, float(value.item()))

    def _trigger_update(self, env) -> Optional[Dict]:
        if self.buffer.size() == 0:
            return None
        bootstrap = {}
        for n, state in self._local_critic_states(env).items():
            state_t = torch.as_tensor(state, dtype=torch.float32,
                                      device=self.trainer.device)
            with torch.no_grad():
                bootstrap[n] = float(self.critic.get_value(state_t).item())
        metrics = self.trainer.update(self.buffer, bootstrap)
        self.buffer.clear()
        return metrics

    def run_step(self, env):
        rewards, done, info = super().run_step(env)
        self._attach_reward(rewards, info)
        return rewards, done, info


class ComparisonMADDPGPolicy(ComparisonRewardMixin, MADDPGDoDPolicy):
    def run_step(self, env):
        rewards, done, info = super().run_step(env)
        self._attach_reward(rewards, info)
        return rewards, done, info


class _AgentQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, state):
        return self.net(state)


class _MonotonicMixer(nn.Module):
    """State-conditioned non-negative weights guarantee QMIX monotonicity."""

    def __init__(self, global_dim: int, max_agents: int, hidden: int = 128):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(global_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, max_agents))
        self.bias = nn.Sequential(nn.Linear(global_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, agent_q: torch.Tensor, global_state: torch.Tensor):
        k = agent_q.numel()
        weights = F.softplus(self.weight(global_state).flatten()[:k])
        return (weights * agent_q.flatten()).sum() / math.sqrt(max(k, 1)) \
            + self.bias(global_state).flatten()[0]


class QMIXPolicy(ComparisonRewardMixin):
    """Discrete QMIX for the variable set of task decisions in each slot."""

    needs_training = True

    def __init__(self, config, name: str = "QMIX", seed: int = 42):
        self.cfg, self.name = config, name
        self.device = torch.device("cpu")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.state_dim = config.get_state_dim()
        self.action_dim = config.get_action_dim()
        self.global_dim = config.GLOBAL_SUMMARY_DIM
        # Forwarded queues may temporarily exceed MAX_DISPATCH; reserve a
        # conservative task-decision ceiling for the variable-agent mixer.
        self.max_agents = config.N_SATS * 32
        self.q = _AgentQ(self.state_dim, self.action_dim).to(self.device)
        self.q_target = _AgentQ(self.state_dim, self.action_dim).to(self.device)
        self.mixer = _MonotonicMixer(self.global_dim, self.max_agents).to(self.device)
        self.mixer_target = _MonotonicMixer(self.global_dim, self.max_agents).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.q.parameters()) + list(self.mixer.parameters()), lr=3e-4)
        self.replay = deque(maxlen=5000)
        self.pending = None
        self.slot_states: List[np.ndarray] = []
        self.slot_actions: List[int] = []
        self.slot_masks: List[np.ndarray] = []
        self.global_state = np.zeros(self.global_dim, np.float32)
        self.epsilon_start, self.epsilon_end = 1.0, 0.05
        self.steps = 0
        self._eval_mode = False
        self.trainer = SimpleNamespace(update_count=0)

    def collect_critic_values(self, env) -> None:
        self.slot_states, self.slot_actions, self.slot_masks = [], [], []
        self.global_state = np.asarray(
            env.constellation.get_global_summary(env.current_slot), np.float32)

    def act_one(self, state: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        feasible = np.flatnonzero(np.asarray(mask) > 0.5)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
            * max(1.0 - self.steps / 2000.0, 0.0)
        if not self._eval_mode and random.random() < epsilon:
            action = int(random.choice(feasible.tolist()))
        else:
            with torch.no_grad():
                q = self.q(torch.as_tensor(state, dtype=torch.float32,
                                           device=self.device)).cpu().numpy()
            q[np.asarray(mask) < 0.5] = -1e9
            action = int(np.argmax(q))
        return action, 0.0

    def get_actions(self, obs, masks):
        return {n: [self.act_one(s, m)[0] for s, m in zip(obs.get(n, []), masks.get(n, []))]
                for n in range(self.cfg.N_SATS)}

    def record_task_transition(self, sat_id, slot_t, state, action,
                               log_prob, mask, task_reward=0.0) -> None:
        if self._eval_mode:
            return
        self.slot_states.append(np.asarray(state, np.float32))
        self.slot_actions.append(int(action))
        self.slot_masks.append(np.asarray(mask, np.float32))

    def _record(self, system_reward: float, done: bool) -> None:
        current = (list(self.slot_states), list(self.slot_actions),
                   list(self.slot_masks), self.global_state.copy())
        if self.pending is not None:
            self.replay.append((*self.pending, current, False))
        if current[0]:
            if done:
                empty = ([], [], [], np.zeros(self.global_dim, np.float32))
                self.replay.append((current, system_reward, empty, True))
                self.pending = None
            else:
                self.pending = (current, system_reward)

    def _mix_value(self, record, target: bool, greedy: bool = False):
        states, actions, masks, global_state = record
        if not states:
            return torch.zeros((), device=self.device)
        st = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        qnet = self.q_target if target else self.q
        q_all = qnet(st)
        if greedy:
            mt = torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=self.device)
            selected = q_all.masked_fill(mt < 0.5, -1e9).max(dim=1).values
        else:
            at = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            selected = q_all.gather(1, at[:, None]).squeeze(1)
        gs = torch.as_tensor(global_state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        mixer = self.mixer_target if target else self.mixer
        return mixer(selected, gs)

    def _update(self) -> None:
        if len(self.replay) < 64:
            return
        batch = random.sample(self.replay, min(32, len(self.replay)))
        losses = []
        for current, reward, nxt, done in batch:
            pred = self._mix_value(current, target=False)
            with torch.no_grad():
                target = torch.tensor(float(reward), device=self.device)
                if not done:
                    target = target + 0.99 * self._mix_value(nxt, target=True, greedy=True)
            losses.append(F.smooth_l1_loss(pred, target))
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(self.q.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()
        if self.steps % 100 == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())

    def run_step(self, env):
        _, rewards, done, info = env.step(policy=self)
        system_reward = float(np.mean(list(rewards.values())))
        info["comparison_system_reward"] = system_reward
        if not self._eval_mode:
            self._record(system_reward, done)
            self.steps += 1
            self._update()
        return rewards, done, info

    def set_train_mode(self):
        self._eval_mode = False; self.q.train(); self.mixer.train()

    def set_eval_mode(self):
        self._eval_mode = True; self.q.eval(); self.mixer.eval()

    def save(self, path: str, extra_info=None):
        os.makedirs(path, exist_ok=True)
        torch.save(self.q.state_dict(), os.path.join(path, "agent_q.pth"))
        torch.save(self.mixer.state_dict(), os.path.join(path, "mixer.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"steps": self.steps, **(extra_info or {})}, f, indent=2)
