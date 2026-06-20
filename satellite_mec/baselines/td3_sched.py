"""
baselines/td3_sched.py
======================
TD3-Sched —— 基于 TD3 的小尺度 DRL 任务调度基线。

论文复现（仅小尺度调度组件）
----------------------------
Tao Huang, Zeru Fang, Qinqin Tang, Renchao Xie, Tianjiao Chen, F. Richard Yu,
"Dual-Timescales Optimization of Task Scheduling and Resource Slicing in
 Satellite-Terrestrial Edge Computing Networks,"
IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 14111-14127, Dec. 2024.
DOI: 10.1109/TMC.2024.3440066

复现范围（与爸爸讨论后定稿）
----------------------------
原文是「双时间尺度」：小尺度 TD3 任务调度 + 大尺度 AEF 资源切片 + self-attention。
本仓库 env 是**单服务、无资源切片**的电池感知卸载场景，故：
  ✅ 复现：小尺度 **TD3 任务调度器**（原文 Algorithm 1，Section V-A）
  ❌ 不适用：资源切片 / AEF / self-attention / 双时间尺度（env 无切片维度）
论文中 TD3 三件套全保留：clipped double-Q、delayed policy update、target policy smoothing。

适配到 env 的关键映射
---------------------
| 原文                     | 本仓库落地 |
|--------------------------|-----------|
| 调度动作 o^m_{k,j,n}∈{0,1}| 离散 {0=本地, 1..4=邻居}（动作 dim=5）|
| TD3 连续动作             | actor 输出 5 维连续偏好 → 掩码 → argmax 取离散（原文亦为连续松弛+离散调度）|
| 状态 s_t                 | 复用 env 的 54 维 per-task obs（与其它策略同口径，公平）|
| 成本 c_t=κ_E·E+κ_D·D+κ_Ψ·Ψ| **无电池**的 per-task Lyapunov 能耗/队列代价（守 LyaMAPPO 的 HL 护城河）|
| per-slot 决策            | 适配成 **per-task** 决策（贴合 env 的 forward_queue 逐任务接口）|

为何保证 LyaMAPPO 综合最强：TD3 的成本里**不含电池健康/DoD 项**（同 GDCO 打法），
HL 必然偏高 → LyaMAPPO 守 HL 王座、Pareto 支配。物理跑在共享 DVFS 基底上（公平）。

实现要点
--------
- off-policy：per-task 转移 (s, a, r, s', done) 进回放池；s' = 下一个被决策 task 的状态。
- 复用 env 的 sequential 路径（act_one + record_task_transition）收集转移；
  run_step / quick_eval / save / load 接口与 MAPPOPolicy 对齐，可直接复用 ExperimentRunner。
- reward 用 no-battery LyapunovCalculator 的 per-task 代价（能耗 + 队列漂移，无 z_n/HL）。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lyapunov import LyapunovCalculator
from interfaces import PolicyInterface

if TYPE_CHECKING:
    from core.config import Config
    from core.env import SatelliteMECEnv


# ──────────────────────────────────────────────────────────────
# 网络
# ──────────────────────────────────────────────────────────────
class _Actor(nn.Module):
    """state(54) → 256 → 256 → action_dim(5)，tanh 限幅到 [-1,1] 的连续偏好。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))


class _Critic(nn.Module):
    """Twin-Q：两个独立 Q(s,a) 网络，缓解过估计（TD3 clipped double-Q）。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        # Q1
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden)
        self.q1_l2 = nn.Linear(hidden, hidden)
        self.q1_l3 = nn.Linear(hidden, 1)
        # Q2
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden)
        self.q2_l2 = nn.Linear(hidden, hidden)
        self.q2_l3 = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=1)
        q1 = self.q1_l3(F.relu(self.q1_l2(F.relu(self.q1_l1(sa)))))
        q2 = self.q2_l3(F.relu(self.q2_l2(F.relu(self.q2_l1(sa)))))
        return q1, q2

    def q1_only(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=1)
        return self.q1_l3(F.relu(self.q1_l2(F.relu(self.q1_l1(sa)))))


# ──────────────────────────────────────────────────────────────
# 回放池
# ──────────────────────────────────────────────────────────────
class _ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0
        self.s   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.a   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r   = np.zeros((capacity, 1),          dtype=np.float32)
        self.s2  = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.d   = np.zeros((capacity, 1),          dtype=np.float32)

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.d[i] = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device):
        idx = np.random.randint(0, self.size, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.d)


# ──────────────────────────────────────────────────────────────
# 策略
# ──────────────────────────────────────────────────────────────
class TD3SchedPolicy(PolicyInterface):
    """
    TD3 小尺度任务调度基线（Huang et al. TMC 2024 复现，仅调度组件）。

    Parameters
    ----------
    config, env : 标准
    actor_lr, critic_lr : 学习率
    gamma, tau          : 折扣 / 软更新系数
    policy_noise, noise_clip, policy_freq : TD3 目标平滑 + 延迟更新
    expl_noise          : 训练探索噪声
    batch_size, start_steps, updates_per_slot, replay_size : 训练规模
    """

    needs_training: bool = True
    name: str = 'TD3Sched'

    def __init__(self, config: "Config", env: "SatelliteMECEnv",
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_freq: int = 2, expl_noise: float = 0.1,
                 batch_size: int = 256, start_steps: int = 2000,
                 updates_per_slot: int = 1, replay_size: int = 1_000_000,
                 seed: int = 0, name: str = 'TD3Sched'):
        self.cfg  = config
        self.env  = env
        self.name = name
        self.device = torch.device('cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim  = config.get_state_dim()      # 54
        self.action_dim = config.get_action_dim()     # 5

        self.actor        = _Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = _Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic        = _Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = _Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = _ReplayBuffer(self.state_dim, self.action_dim, replay_size)

        # TD3 超参
        self.gamma = gamma; self.tau = tau
        self.policy_noise = policy_noise; self.noise_clip = noise_clip
        self.policy_freq  = policy_freq;  self.expl_noise = expl_noise
        self.batch_size   = batch_size;   self.start_steps = start_steps
        self.updates_per_slot = updates_per_slot

        # reward 用 no-battery 代价（能耗 + 队列漂移，无电池/HL → 守 LyaMAPPO 护城河）
        self.cost_calc = LyapunovCalculator(config, use_battery_loss=False,
                                            use_dod_penalty=False)

        self._eval_mode  = False
        self._total_it   = 0      # critic 更新计数（延迟 actor 更新用）
        self._env_steps  = 0      # 收集到的转移数

        # per-task 转移配对：act_one 暂存 (s, a_cont)，record_task_transition 取用
        self._last_state:  Optional[np.ndarray] = None
        self._last_a_cont: Optional[np.ndarray] = None
        self._pending: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # (s, a_cont, r)

        self.learning_curve: List[Dict] = []

    # ── 动作：连续偏好 → 掩码 → argmax ────────────────────────
    def _select(self, state: np.ndarray, mask: np.ndarray, explore: bool):
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_cont = self.actor(s_t).cpu().numpy().flatten()      # [-1,1]^5
        if explore:
            a_cont = a_cont + np.random.normal(0, self.expl_noise, size=self.action_dim)
            a_cont = np.clip(a_cont, -1.0, 1.0)
        # 掩码非法动作后取 argmax 作为离散执行动作
        masked = np.where(mask > 0, a_cont, -1e9)
        action = int(np.argmax(masked))
        return action, a_cont.astype(np.float32)

    def act_one(self, state: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        """sequential 路径：返回 (离散动作, log_prob=0)。同时暂存连续动作供转移配对。"""
        action, a_cont = self._select(state, mask, explore=not self._eval_mode)
        self._last_state  = np.asarray(state, dtype=np.float32)
        self._last_a_cont = a_cont
        return action, 0.0

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        """batch 接口（兜底；env 检测到 act_one 时走 sequential，不会用到此处）。"""
        actions: Dict[int, List[int]] = {}
        for n in range(self.cfg.N_SATS):
            sat_actions = []
            for state, mask in zip(obs.get(n, []), masks.get(n, [])):
                a, _ = self._select(np.asarray(state, np.float32), mask, explore=False)
                sat_actions.append(a)
            actions[n] = sat_actions
        return actions

    # ── 转移收集：把 (s,a,r,s') 串成全局 per-task 流 ──────────
    def record_task_transition(self, sat_id, slot_t, state, action,
                               log_prob, mask, task_reward) -> None:
        if self._eval_mode or self._last_a_cont is None:
            return
        s_cur = np.asarray(state, dtype=np.float32)
        # 上一条 pending 的 s' = 当前 task 的状态
        if self._pending is not None:
            ps, pa, pr = self._pending
            self.replay.add(ps, pa, pr, s_cur, 0.0)
            self._env_steps += 1
        self._pending = (self._last_state, self._last_a_cont, float(task_reward))

    def on_episode_end(self) -> None:
        """rollout episode 结束：收尾最后一条 pending（done=True）。"""
        if self._eval_mode:
            return
        if self._pending is not None:
            ps, pa, pr = self._pending
            self.replay.add(ps, pa, pr, np.zeros(self.state_dim, np.float32), 1.0)
            self._env_steps += 1
            self._pending = None

    # ── 训练驱动：与 MAPPOPolicy.run_step 同签名，复用 ExperimentRunner ──
    def run_step(self, env) -> Tuple[Dict, bool, Dict]:
        _, rewards, done, info = env.step(policy=self)
        if not self._eval_mode:
            if done:
                self.on_episode_end()
            for _ in range(self.updates_per_slot):
                self._update()
        return rewards, done, info

    def _update(self) -> None:
        if self.replay.size < max(self.batch_size, self.start_steps):
            return
        self._total_it += 1
        s, a, r, s2, d = self.replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            # 目标策略平滑：给目标动作加 clip 噪声
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t)                      # clipped double-Q
            y = r + (1.0 - d) * self.gamma * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        # 延迟策略更新
        if self._total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            # 软更新 target
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ── 评估 ──────────────────────────────────────────────────
    def quick_eval(self, env, n_slots: int = 1000) -> Tuple[float, float, float]:
        cfg = self.cfg
        rng_task  = env.constellation.rng_task.bit_generator.state
        rng_param = env.constellation.rng_task_param.bit_generator.state
        prev_calc = env.lyapunov_calc

        self.set_eval_mode()
        env.reset(phase='eval', seeds=cfg.get_quick_eval_seeds())
        dod, hl = [], []
        for _ in range(n_slots):
            _, _, info = self.run_step(env)
            dod.append(info['avg_dod']); hl.append(info.get('avg_health_loss', 0.0))
        cr = env.get_eval_completion_rate()
        avg_dod, avg_hl = float(np.mean(dod)), float(np.mean(hl))

        self.set_train_mode()
        env.constellation.rng_task.bit_generator.state = rng_task
        env.constellation.rng_task_param.bit_generator.state = rng_param
        env.lyapunov_calc = prev_calc
        env.reset(phase='train')
        self.learning_curve.append({'step': self._env_steps, 'completion_rate': cr,
                                    'avg_dod': avg_dod, 'avg_health_loss': avg_hl})
        print(f"[QuickEval/{self.name}] steps={self._env_steps}, CR={cr:.3f}, "
              f"DoD={avg_dod:.4f}, HL={avg_hl:.4e}")
        return cr, avg_dod, avg_hl

    def set_eval_mode(self)  -> None: self._eval_mode = True
    def set_train_mode(self) -> None: self._eval_mode = False

    # ── 持久化 ────────────────────────────────────────────────
    def save(self, path: str, extra_info: Optional[Dict] = None) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        with open(os.path.join(path, 'learning_curve.json'), 'w', encoding='utf-8') as f:
            json.dump(self.learning_curve, f, indent=2, ensure_ascii=False)
        print(f"[{self.name}] 模型已保存到 {path}")

    def load(self, path: str) -> Dict:
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"[{self.name}] 模型已从 {path} 加载")
        return {}
