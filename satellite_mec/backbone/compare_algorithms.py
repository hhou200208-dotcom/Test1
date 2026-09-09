"""Algorithms used by the RL-backbone comparison.

Design goals
------------
* identical SatelliteMECEnv and dynamic action masks;
* identical system reward R_sys = mean_n r_n for learning/evaluation;
* BETA_TASK=0 for PPO backbones (no BLA task-credit refinement);
* MAPPO vs IPPO differs in critic information only;
* MADDPG reuses the repository's discrete Gumbel-Softmax learner but receives R_sys;
* QMIX uses a decision-round adapter: the k-th task decision of every satellite in a
  physical slot forms one MARL micro-step. Inactive satellites take an internal NOOP.
  No physical time elapses inside a slot, so within-slot TD discount is 1; the last
  micro-step of a slot uses the physical-slot discount gamma.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from interfaces import PolicyInterface
from training import MAPPOPolicy
from training.trainer import MAPPOTrainer
from baselines.maddpg_dod import MADDPGDoDPolicy


def system_reward(rewards: Dict[int, float]) -> float:
    """Shared cooperative reward used by every backbone."""
    return float(np.mean(list(rewards.values()))) if rewards else 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def module_finite(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in module.parameters())


class SharedRewardMAPPO(MAPPOPolicy):
    """Vanilla MAPPO backbone with a shared system-level slot reward."""

    def store_slot_data(self, t: int, rewards: Dict[int, float],
                        n_executed: Dict[int, int], done: bool) -> None:
        r_sys = system_reward(rewards)
        for n in range(self.cfg.N_SATS):
            if n in self._slot_critic:
                critic_state, value = self._slot_critic[n]
                self.buffer.add_slot(n, t, critic_state, r_sys, value, done)
        # Sequential env path already calls record_task_transition() task by task.
        # Keep the batch fallback for compatibility.
        for n in range(self.cfg.N_SATS):
            n_exec = n_executed.get(n, 0)
            for state, action, log_prob, mask in self._slot_inference.get(n, [])[:n_exec]:
                self.buffer.add_task(n, t, state, action, log_prob, mask)


class _LocalCritic(nn.Module):
    """IPPO critic: own task-free node state only (47 dims for current config)."""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.get_state_dim() - 7
        h = config.HIDDEN_DIM
        self.norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        y = self.net(self.norm(x)).squeeze(-1)
        return y.squeeze(0) if squeeze else y


class _IPPOTrainer(MAPPOTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.critic = _LocalCritic(config).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.LR_CRITIC)

    def compute_bootstrap(self, env, buffer) -> Dict[int, float]:
        d = self.cfg.get_state_dim() - 7
        out: Dict[int, float] = {}
        for sat_id, critic_state in env.get_critic_obs().items():
            local = torch.as_tensor(np.asarray(critic_state[:d], np.float32),
                                    device=self.device)
            with torch.no_grad():
                out[sat_id] = float(self.critic.get_value(local).item())
        return out


class IPPOPolicy(SharedRewardMAPPO):
    """Independent PPO control: same actor/PPO settings as MAPPO, local critic only."""

    def __init__(self, config, name: str = "IPPO"):
        super().__init__(config, name=name)
        # Replace centralized trainer created by the parent with the local-critic trainer.
        self.trainer = _IPPOTrainer(config)
        self.actor = self.trainer.actor
        self.critic = self.trainer.critic

    def collect_critic_values(self, env) -> None:
        d = self.cfg.get_state_dim() - 7
        self._slot_critic = {}
        for n, critic_state in env.get_critic_obs().items():
            local = np.asarray(critic_state[:d], np.float32)
            x = torch.as_tensor(local, device=self.trainer.device)
            with torch.no_grad():
                value = float(self.critic.get_value(x).item())
            self._slot_critic[n] = (local, value)


class SharedRewardMADDPG(MADDPGDoDPolicy):
    """Repository discrete MADDPG learner, with the common R_sys substituted for reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_update_count = 0

    def _flush_slot(self, rewards: Dict, done: bool,
                    info: Optional[Dict] = None) -> None:
        if not self._slot_tasks:
            return
        r_sys = system_reward(rewards) * self.reward_scale
        # Preserve the repository learner's one-step transition convention while making
        # the reward exactly shared across satellites/tasks.
        for (s, cobs, a, mask, _action_cost, _sat_id) in self._slot_tasks:
            if self._pending is not None:
                ps, pc, pa, pr = self._pending
                self.replay.add(ps, pc, pa, pr, s, cobs, mask, 0.0)
                self._env_steps += 1
            self._pending = (s, cobs, a, r_sys)
        # Rollout boundaries are training truncations in the physical simulator.  The
        # original learner marks them terminal; keep that convention for this pilot and
        # report it explicitly in metadata.  We can remove it in the final sensitivity run.
        if done and self._pending is not None:
            ps, pc, pa, pr = self._pending
            self.replay.add(ps, pc, pa, pr,
                            np.zeros(self.s_dim, np.float32),
                            np.zeros(self.c_dim, np.float32),
                            np.ones(self.a_dim, np.float32), 1.0)
            self._env_steps += 1
            self._pending = None

    def _update(self) -> None:
        ready = self.replay.size >= max(self.batch_size, self.start_steps)
        super()._update()
        if ready:
            self.backbone_update_count += 1


class _AgentQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(obs))


class _Mixer(nn.Module):
    """Monotonic QMIX mixer with state-conditioned non-negative weights."""

    def __init__(self, global_dim: int, n_agents: int, hidden: int):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents * hidden),
        )
        self.hyper_b1 = nn.Linear(global_dim, hidden)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, agent_q: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        # agent_q [B,N], global_state [B,G]
        b = agent_q.shape[0]
        w1 = F.softplus(self.hyper_w1(global_state)).view(b, self.n_agents, -1)
        b1 = self.hyper_b1(global_state).unsqueeze(1)
        hidden = F.elu(torch.bmm(agent_q.unsqueeze(1), w1) + b1)
        w2 = F.softplus(self.hyper_w2(global_state)).unsqueeze(-1)
        b2 = self.hyper_b2(global_state).squeeze(-1)
        return torch.bmm(hidden, w2).squeeze(-1).squeeze(-1) + b2


class _MicroReplay:
    """Memory-conscious replay; observations/globals are stored as float16."""

    def __init__(self, capacity: int):
        self.data = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.data)

    def add(self, obs, mask, action, g, reward, discount,
            next_obs, next_mask, next_g):
        self.data.append((
            np.asarray(obs, np.float16), np.asarray(mask, np.bool_),
            np.asarray(action, np.int8), np.asarray(g, np.float16),
            np.float32(reward), np.float32(discount),
            np.asarray(next_obs, np.float16), np.asarray(next_mask, np.bool_),
            np.asarray(next_g, np.float16),
        ))

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.data), size=batch)
        rows = [self.data[int(i)] for i in idx]
        cols = list(zip(*rows))
        return tuple(np.stack(c, axis=0) for c in cols)


class MicroStepQMIXPolicy(PolicyInterface):
    """QMIX adapted to the repository's sequential per-task scheduler.

    The environment remains physically unchanged.  During each physical slot, the k-th
    task decision of every satellite is grouped as decision round k.  Satellites without
    a k-th task take internal NOOP (action index env_action_dim).  This preserves the
    repository's task-level action semantics while exposing a standard cooperative joint
    action to the QMIX mixer.
    """

    needs_training = True
    name = "QMIX"

    def __init__(self, config, seed: int = 0, lr: float = 5e-4,
                 batch_size: int = 32, replay_size: int = 5000,
                 start_transitions: int = 256, target_interval: int = 200,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_anneal_slots: int = 8000):
        self.cfg = config
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.n_agents = config.N_SATS
        self.obs_dim = config.get_state_dim()
        self.env_action_dim = config.get_action_dim()
        self.action_dim = self.env_action_dim + 1  # internal NOOP
        self.noop = self.env_action_dim
        self.global_dim = config.get_critic_state_dim()
        h = config.HIDDEN_DIM

        self.agent_q = _AgentQ(self.obs_dim, self.action_dim, h).to(self.device)
        self.target_agent_q = _AgentQ(self.obs_dim, self.action_dim, h).to(self.device)
        self.mixer = _Mixer(self.global_dim, self.n_agents, h).to(self.device)
        self.target_mixer = _Mixer(self.global_dim, self.n_agents, h).to(self.device)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()), lr=lr)

        self.batch_size = int(batch_size)
        self.start_transitions = int(start_transitions)
        self.target_interval = int(target_interval)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_anneal_slots = int(eps_anneal_slots)
        self.replay = _MicroReplay(replay_size)

        self._eval_mode = False
        self._physical_slots = 0
        self._update_count = 0
        self._invalid_actions = 0
        self._global_state = np.zeros(self.global_dim, np.float32)
        self._slot_tasks: Dict[int, List[Tuple[np.ndarray, int, np.ndarray]]] = {}
        self._pending_last = None
        self.last_loss = float("nan")

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def invalid_actions(self) -> int:
        return self._invalid_actions

    def epsilon(self) -> float:
        frac = min(self._physical_slots / max(self.eps_anneal_slots, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        self.agent_q.eval(); self.mixer.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.agent_q.train(); self.mixer.train()

    def collect_critic_values(self, env) -> None:
        cobs = env.get_critic_obs()
        if cobs:
            self._global_state = np.mean(
                np.stack([np.asarray(v, np.float32) for v in cobs.values()]), axis=0)
        else:
            self._global_state = np.zeros(self.global_dim, np.float32)
        self._slot_tasks = {n: [] for n in range(self.n_agents)}

    def act_one(self, state: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        s = torch.as_tensor(np.asarray(state, np.float32), device=self.device).unsqueeze(0)
        env_mask = np.asarray(mask, np.float32)
        feasible = np.flatnonzero(env_mask > 0.5)
        if len(feasible) == 0:
            return 0, 0.0
        with torch.no_grad():
            q = self.agent_q(s).squeeze(0).cpu().numpy()
        if (not self._eval_mode) and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(feasible))
        else:
            q[self.env_action_dim:] = -1e9
            q[:self.env_action_dim][env_mask <= 0.5] = -1e9
            action = int(np.argmax(q))
        if action >= self.env_action_dim or env_mask[action] <= 0.5:
            self._invalid_actions += 1
            action = int(feasible[0])
        return action, 0.0

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        return {n: [self.act_one(s, m)[0]
                    for s, m in zip(obs.get(n, []), masks.get(n, []))]
                for n in range(self.n_agents)}

    def record_task_transition(self, sat_id: int, slot_t: int, state: np.ndarray,
                               action: int, log_prob: float, mask: np.ndarray,
                               task_reward: float = 0.0) -> None:
        if self._eval_mode:
            return
        self._slot_tasks.setdefault(int(sat_id), []).append((
            np.asarray(state, np.float32), int(action), np.asarray(mask, np.float32)))

    def _rounds_from_slot(self):
        max_rounds = max((len(v) for v in self._slot_tasks.values()), default=0)
        max_rounds = max(max_rounds, 1)  # all-NOOP round if no task exists
        rounds = []
        for k in range(max_rounds):
            obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
            masks = np.zeros((self.n_agents, self.action_dim), np.bool_)
            actions = np.full(self.n_agents, self.noop, np.int64)
            masks[:, self.noop] = True
            for n in range(self.n_agents):
                seq = self._slot_tasks.get(n, [])
                if k < len(seq):
                    s, a, m = seq[k]
                    obs[n] = s
                    masks[n, :] = False
                    masks[n, :self.env_action_dim] = m > 0.5
                    actions[n] = int(a)
                    if not masks[n, actions[n]]:
                        self._invalid_actions += 1
                        feasible = np.flatnonzero(m > 0.5)
                        actions[n] = int(feasible[0]) if len(feasible) else self.noop
            rounds.append((obs, masks, actions))
        return rounds

    def _push_transition(self, cur, cur_g, reward, discount, nxt, nxt_g):
        obs, mask, action = cur
        nobs, nmask, _naction = nxt
        self.replay.add(obs, mask, action, cur_g, reward, discount,
                        nobs, nmask, nxt_g)

    def _flush_slot(self, rewards: Dict[int, float]) -> None:
        rounds = self._rounds_from_slot()
        g = self._global_state.copy()
        # Complete the previous physical slot using this slot's first decision round.
        if self._pending_last is not None:
            prev_round, prev_g, prev_reward = self._pending_last
            self._push_transition(prev_round, prev_g, prev_reward,
                                  self.cfg.GAMMA, rounds[0], g)
            self._pending_last = None
        # No physical time passes between decision rounds inside one slot.
        for k in range(len(rounds) - 1):
            self._push_transition(rounds[k], g, 0.0, 1.0, rounds[k + 1], g)
        # Last round receives R_sys and is connected to next physical slot lazily.
        self._pending_last = (rounds[-1], g, system_reward(rewards))

    def finalize_training(self) -> None:
        if self._pending_last is None:
            return
        cur, g, reward = self._pending_last
        zero_obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
        zero_mask = np.zeros((self.n_agents, self.action_dim), np.bool_)
        zero_mask[:, self.noop] = True
        zero_action = np.full(self.n_agents, self.noop, np.int64)
        self._push_transition(cur, g, reward, 0.0,
                              (zero_obs, zero_mask, zero_action),
                              np.zeros(self.global_dim, np.float32))
        self._pending_last = None

    def _update(self) -> None:
        if len(self.replay) < max(self.batch_size, self.start_transitions):
            return
        obs, mask, action, g, reward, discount, nobs, nmask, ng = \
            self.replay.sample(self.batch_size, self.rng)
        obs_t = torch.as_tensor(obs.astype(np.float32), device=self.device)
        mask_t = torch.as_tensor(mask, device=self.device)
        act_t = torch.as_tensor(action.astype(np.int64), device=self.device)
        g_t = torch.as_tensor(g.astype(np.float32), device=self.device)
        r_t = torch.as_tensor(reward.astype(np.float32), device=self.device)
        disc_t = torch.as_tensor(discount.astype(np.float32), device=self.device)
        nobs_t = torch.as_tensor(nobs.astype(np.float32), device=self.device)
        nmask_t = torch.as_tensor(nmask, device=self.device)
        ng_t = torch.as_tensor(ng.astype(np.float32), device=self.device)

        q_all = self.agent_q(obs_t)
        chosen_q = q_all.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_q, g_t)

        with torch.no_grad():
            # Double-Q action selection with target evaluation.
            online_next = self.agent_q(nobs_t).masked_fill(~nmask_t, -1e9)
            next_action = online_next.argmax(dim=-1)
            target_next_all = self.target_agent_q(nobs_t).masked_fill(~nmask_t, -1e9)
            target_next = target_next_all.gather(
                -1, next_action.unsqueeze(-1)).squeeze(-1)
            target_tot = self.target_mixer(target_next, ng_t)
            td_target = r_t + disc_t * target_tot

        loss = F.smooth_l1_loss(q_tot, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()
        self._update_count += 1
        self.last_loss = float(loss.item())
        if not math.isfinite(self.last_loss):
            raise RuntimeError("QMIX produced non-finite loss")

    def _hard_target_update(self) -> None:
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def run_step(self, env):
        _, rewards, done, info = env.step(policy=self)
        if not self._eval_mode:
            self._flush_slot(rewards)
            self._physical_slots += 1
            self._update()
            if self._physical_slots % max(self.target_interval, 1) == 0:
                self._hard_target_update()
        return rewards, done, info

    def parameters_finite(self) -> bool:
        return module_finite(self.agent_q) and module_finite(self.mixer)
