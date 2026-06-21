"""
training/networks.py
====================
神经网络组件：Actor、Critic 以及训练所需的数据结构。

类层次
------
    Actor          : 策略网络（状态 → 动作概率）
    Critic         : 价值网络（全局状态 → 状态价值）
    SlotRecord     : 单时隙经验（用于 Critic 训练）
    TaskRecord     : 单任务经验（用于 Actor 训练）
    RolloutBuffer  : 轨迹缓冲区（存储 + GAE 计算）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ──────────────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    策略网络。

    网络结构
    --------
        LayerNorm → Linear(state_dim, hidden) → ReLU
                  → Linear(hidden, hidden)     → ReLU
                  → Linear(hidden, action_dim)
        + 非法动作掩码（logit → −1e9）

    参数
    ----
    config : Config
        全局配置对象，提供 state_dim / action_dim / HIDDEN_DIM。

    示例
    ----
    >>> actor = Actor(cfg)
    >>> probs = actor.forward(state_tensor, mask_tensor)
    >>> action, log_prob, entropy = actor.get_action(state_t, mask_t, deterministic=False)
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        state_dim  = config.get_state_dim()
        action_dim = config.get_action_dim()
        hidden_dim = config.HIDDEN_DIM

        self.input_norm = nn.LayerNorm(state_dim)
        self.fc1 = nn.Linear(state_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeeze = state.dim() == 1
        if squeeze:
            state, mask = state.unsqueeze(0), mask.unsqueeze(0)
        x = self.input_norm(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        logits = logits + (1.0 - mask) * (-1e9)   # 屏蔽非法动作
        probs  = F.softmax(logits, dim=-1)
        return probs.squeeze(0) if squeeze else probs

    def get_action(
        self,
        state: torch.Tensor,
        mask:  torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        从当前策略中采样（或贪婪选取）一个动作。

        Returns
        -------
        (action, log_prob, normalized_entropy)
        """
        with torch.no_grad():
            probs = self.forward(state, mask)
        if deterministic:
            action = int(torch.argmax(probs).item())
        else:
            action = int(Categorical(probs).sample().item())
        log_prob = float(torch.log(probs[action] + 1e-8).item())
        entropy  = float(self._compute_norm_entropy(probs, mask).item())
        return action, log_prob, entropy

    def evaluate_action(
        self,
        states:  torch.Tensor,
        masks:   torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量计算动作的 log_prob 和归一化熵（用于 PPO 更新）。"""
        probs = self.forward(states, masks)
        log_probs  = torch.log(probs.gather(1, actions.long().unsqueeze(1)).squeeze(1) + 1e-8)
        entropies  = self._compute_norm_entropy(probs, masks)
        return log_probs, entropies

    def _compute_norm_entropy(
        self,
        probs: torch.Tensor,
        mask:  torch.Tensor,
    ) -> torch.Tensor:
        n_feasible  = mask.sum(dim=-1)
        log_probs   = torch.log(probs + 1e-8)
        entropy     = -(probs * log_probs * mask).sum(dim=-1)
        max_entropy = torch.log(n_feasible.float() + 1e-8)
        return entropy / (max_entropy + 1e-8)

    def get_probs_numpy(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probs = self.forward(torch.FloatTensor(state), torch.FloatTensor(mask))
        return probs.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────
class Critic(nn.Module):
    """
    中心化价值网络（CTDE 架构）。

    输入为全局 critic 状态（维度 = critic_state_dim = 135），
    输出为标量状态价值。

    参数
    ----
    config : Config
        提供 critic_state_dim / HIDDEN_DIM。
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        critic_dim = config.get_critic_state_dim()
        hidden_dim = config.HIDDEN_DIM

        self.input_norm = nn.LayerNorm(critic_dim)
        self.fc1 = nn.Linear(critic_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, critic_state: torch.Tensor) -> torch.Tensor:
        squeeze = critic_state.dim() == 1
        if squeeze:
            critic_state = critic_state.unsqueeze(0)
        x = self.input_norm(critic_state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(0) if squeeze else value

    def get_value(self, critic_state: torch.Tensor) -> torch.Tensor:
        return self.forward(critic_state).squeeze(-1)

    def get_value_numpy(self, critic_state: np.ndarray) -> float:
        with torch.no_grad():
            value = self.get_value(torch.FloatTensor(critic_state))
        return float(value.item())


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SlotRecord:
    """单时隙的 Critic 经验记录。"""
    sat_id:       int
    slot_t:       int
    critic_state: np.ndarray
    slot_reward:  float
    value:        float
    done:         bool
    advantage:    float = 0.0
    target_return:float = 0.0


@dataclass
class TaskRecord:
    """单任务的 Actor 经验记录。"""
    sat_id:   int
    slot_t:   int
    state:    np.ndarray
    action:   int
    log_prob: float
    mask:     np.ndarray
    advantage:   float = 0.0
    task_reward: float = 0.0   # Stage 6：任务级即时奖励（apply_action 返回）


# ──────────────────────────────────────────────────────────────────────────────
# RolloutBuffer
# ──────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    轨迹经验缓冲区。

    职责
    ----
    1. 存储 SlotRecord（Critic 用）和 TaskRecord（Actor 用）
    2. 使用 GAE 计算 advantage 和 target_return
    3. 组装成 mini-batch 供 PPO 更新

    使用示例
    --------
    >>> buf = RolloutBuffer(cfg)
    >>> buf.add_slot(...)
    >>> buf.add_task(...)
    >>> buf.compute_gae(bootstrap_values, gamma=0.99, lambda_gae=0.95)
    >>> actor_batch  = buf.get_actor_batch()
    >>> critic_batch = buf.get_critic_batch()
    >>> buf.clear()
    """

    def __init__(self, config):
        self.cfg = config
        self._slot_records:  List[SlotRecord]  = []
        self._task_records:  List[TaskRecord]  = []
        self._slot_index:    Dict[Tuple, SlotRecord] = {}

    def add_slot(
        self,
        sat_id:       int,
        slot_t:       int,
        critic_state: np.ndarray,
        slot_reward:  float,
        value:        float,
        done:         bool,
    ) -> None:
        r = SlotRecord(
            sat_id=sat_id, slot_t=slot_t,
            critic_state=critic_state.copy(),
            slot_reward=slot_reward, value=value, done=done,
        )
        self._slot_records.append(r)
        self._slot_index[(sat_id, slot_t)] = r

    def add_task(
        self,
        sat_id:      int,
        slot_t:      int,
        state:       np.ndarray,
        action:      int,
        log_prob:    float,
        mask:        np.ndarray,
        task_reward: float = 0.0,
    ) -> None:
        self._task_records.append(TaskRecord(
            sat_id=sat_id, slot_t=slot_t,
            state=state.copy(), action=action,
            log_prob=log_prob, mask=mask.copy(),
            task_reward=task_reward,
        ))

    def compute_gae(
        self,
        bootstrap_values: Dict[int, float],
        gamma:            float,
        lambda_gae:       float,
    ) -> None:
        """
        计算广义优势估计（GAE）。

        Parameters
        ----------
        bootstrap_values : {sat_id: V(s_T)}，轨迹末尾的 bootstrap 价值
        gamma            : 折扣因子
        lambda_gae       : GAE λ 参数
        """
        # 按卫星分组并按时间排序
        slots_by_sat: Dict[int, List[SlotRecord]] = {}
        for r in self._slot_records:
            slots_by_sat.setdefault(r.sat_id, []).append(r)

        for sat_id, slots in slots_by_sat.items():
            slots.sort(key=lambda r: r.slot_t)
            next_value = bootstrap_values.get(sat_id, 0.0)
            gae = 0.0
            for r in reversed(slots):
                if r.done:
                    next_value = 0.0
                    gae = 0.0
                delta = r.slot_reward + gamma * next_value - r.value
                gae   = delta + gamma * lambda_gae * gae
                r.advantage    = gae
                r.target_return = gae + r.value
                next_value = r.value

        # Stage 6：per-task advantage = slot_advantage + β · (task_r − slot_mean_task_r)
        # 同一 (sat, slot) 内任务 task_reward 的相对偏差给出局部信用
        beta_task = self.cfg.BETA_TASK
        # 按 (sat, slot) 分组算平均 task_reward
        tasks_by_slot: Dict[Tuple[int, int], List[TaskRecord]] = {}
        for tr in self._task_records:
            tasks_by_slot.setdefault((tr.sat_id, tr.slot_t), []).append(tr)
        for (sat_id, slot_t), trs in tasks_by_slot.items():
            sr = self._slot_index.get((sat_id, slot_t))
            slot_adv = sr.advantage if sr else 0.0
            rewards = np.array([t.task_reward for t in trs], dtype=np.float32)
            mean_r = float(rewards.mean()) if len(rewards) > 0 else 0.0
            std_r  = float(rewards.std())  if len(rewards) > 1 else 1.0
            std_r  = max(std_r, 1e-6)
            for t in trs:
                local = (t.task_reward - mean_r) / std_r
                t.advantage = slot_adv + beta_task * local

        self._normalize_advantages()

    def _normalize_advantages(self) -> None:
        if not self._task_records:
            return
        adv  = np.array([r.advantage for r in self._task_records], dtype=np.float32)
        mean, std = adv.mean(), adv.std()
        for r in self._task_records:
            r.advantage = (r.advantage - mean) / (std + 1e-8)

    def get_actor_batch(self) -> Dict[str, np.ndarray]:
        """返回 Actor 更新所需的批数据（dict of numpy arrays）。"""
        if not self._task_records:
            return {}
        return {
            'states':        np.array([r.state    for r in self._task_records], np.float32),
            'actions':       np.array([r.action   for r in self._task_records], np.int64),
            'log_probs_old': np.array([r.log_prob for r in self._task_records], np.float32),
            'masks':         np.array([r.mask     for r in self._task_records], np.float32),
            'advantages':    np.array([r.advantage for r in self._task_records], np.float32),
        }

    def get_critic_batch(self) -> Dict[str, np.ndarray]:
        """返回 Critic 更新所需的批数据（dict of numpy arrays）。"""
        if not self._slot_records:
            return {}
        return {
            'critic_states':  np.array([r.critic_state  for r in self._slot_records], np.float32),
            'target_returns': np.array([r.target_return for r in self._slot_records], np.float32),
            'values_old':     np.array([r.value         for r in self._slot_records], np.float32),
        }

    def clear(self) -> None:
        self._slot_records.clear()
        self._task_records.clear()
        self._slot_index.clear()

    def size(self)      -> int: return len(self._task_records)
    def slot_size(self) -> int: return len(self._slot_records)
    def __len__(self)   -> int: return self.size()
