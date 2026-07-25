"""
baselines/maddpg_dod.py
=======================
MADDPG offloading baseline —— 遵循 Zhong et al. IoT-J 2026 的 MADDPG 卸载方法。

对比设定（与爸爸讨论后定稿，公平起跑线口径）
--------------------------------------------
Liang Zhong, Shen Tian, Deze Zeng, Zhihao Qu, Chengyu Hu,
"Battery Lifetime Extension in Heterogeneous Satellite Edge Computing:
 A Lyapunov-DRL Approach," IEEE Internet of Things Journal, 2026.

本 baseline = 论文核心学习器 **MADDPG（Gumbel-Softmax 离散 actor + 中心化 critic,
CTDE）**,在**与所有学习型基线相同的环境 / DVFS 物理基底 / 完成激励**下训练,以保证
对比公平。它相对 LyaMAPPO 的差别仅在于:
  - off-policy MADDPG 学习器 vs on-policy MAPPO;
  - **缺少 LyaMAPPO 的电池虚拟队列 z_n（HL 半衰期感知）与任务级优势分解**。
因此其电池健康(HL)显著更差 —— 这正是 LyaMAPPO 的命门优势。

奖励口径（忠实论文的电池处理:DoD-aware,无 HL）
------------------------------------------------
- 使用 env 的 outcome-aware 奖励（含完成/超时/拒绝/队列激励 → 不塌吞吐,公平起跑线）;
- **保留 DoD 感知**（LyapunovCalculator use_dod_penalty=True,对齐论文 υ·D 目标）;
- **不含 HL 半衰期项**（use_battery_loss=False,且训练脚本置 W_HL=0）—— 论文无此项,
  HL 半衰期虚拟队列是 LyaMAPPO 原创。
（上述在 train_maddpg_dod.py 中装配 env.lyapunov_calc 与 W_HL。）

架构（对齐论文 Table II + 本仓库 CTDE 基建）
--------------------------------------------
- Actor（去中心化执行）: 54 维局部态 → 256 → 256 → 5 logits → 掩码 → Gumbel-Softmax;
- Critic（中心化训练）: 复用本仓库可扩展 245 维 critic 态（局部 5 节点 + 全局摘要）
  Q(o^critic_245, a_5) → 256 → 256 → 1;目标网软更新 σ=0.01;Adam;LR 1e-3。
- off-policy 每任务转移,复用 env 的 sequential 路径（act_one + record_task_transition），
  run_step/quick_eval/save/load 与 TD3SchedPolicy/MAPPOPolicy 对齐,直接复用 eval_*。
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
# 网络（Table II：Actor/Critic 均 Input + 2×256 Hidden + Output, ReLU）
# ──────────────────────────────────────────────────────────────
class _Actor(nn.Module):
    """去中心化 Actor：局部态(54) → 256 → 256 → logits(5)。掩码 + Gumbel-Softmax 在外部施加。"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return self.l3(x)                       # logits（未 softmax）


class _Critic(nn.Module):
    """中心化 Critic：Q(critic_state_245, action_5) → 256 → 256 → 1。"""

    def __init__(self, critic_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(critic_dim + action_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c, a], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# ──────────────────────────────────────────────────────────────
# 回放池（每任务转移，含中心化 critic 态）
# ──────────────────────────────────────────────────────────────
class _ReplayBuffer:
    def __init__(self, s_dim: int, c_dim: int, a_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity; self.ptr = 0; self.size = 0
        self.s   = np.zeros((capacity, s_dim), dtype=np.float32)
        self.c   = np.zeros((capacity, c_dim), dtype=np.float32)
        self.a   = np.zeros((capacity, a_dim), dtype=np.float32)
        self.r   = np.zeros((capacity, 1),     dtype=np.float32)
        self.s2  = np.zeros((capacity, s_dim), dtype=np.float32)
        self.c2  = np.zeros((capacity, c_dim), dtype=np.float32)
        self.m2  = np.zeros((capacity, a_dim), dtype=np.float32)
        self.d   = np.zeros((capacity, 1),     dtype=np.float32)

    def add(self, s, c, a, r, s2, c2, m2, done):
        i = self.ptr
        self.s[i] = s; self.c[i] = c; self.a[i] = a; self.r[i] = r
        self.s2[i] = s2; self.c2[i] = c2; self.m2[i] = m2; self.d[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device):
        idx = np.random.randint(0, self.size, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return (to(self.s), to(self.c), to(self.a), to(self.r),
                to(self.s2), to(self.c2), to(self.m2), to(self.d))


# ──────────────────────────────────────────────────────────────
# 策略
# ──────────────────────────────────────────────────────────────
class MADDPGDoDPolicy(PolicyInterface):
    """MADDPG 卸载 baseline（Zhong et al. IoT-J 2026 学习器,公平起跑线口径）。

    Parameters
    ----------
    config, env : 标准
    actor_lr, critic_lr : 学习率（Table II：均 1e-3）
    gamma, tau  : 折扣 / 目标网软更新（Table II：σ=0.01）
    tau_gs_start, tau_gs_end, anneal_slots : Gumbel-Softmax 温度退火
    expl_noise  : 训练期 logits 高斯噪声（对应原文 N(0,1) 探索）
    batch_size, start_steps, updates_per_slot, replay_size, reward_scale
    """

    needs_training: bool = True
    name: str = 'MADDPG_DoD'

    def __init__(self, config: "Config", env: "SatelliteMECEnv",
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01,
                 tau_gs_start: float = 1.0, tau_gs_end: float = 0.5,
                 anneal_slots: int = 30000, expl_noise: float = 0.2,
                 batch_size: int = 256, start_steps: int = 2000,
                 updates_per_slot: int = 1, replay_size: int = 1_000_000,
                 reward_scale: float = 0.1,
                 zhong_reward: bool = False, upsilon: Optional[float] = None,
                 zw_q: float = 1.0, zw_y: float = 1.0,
                 seed: int = 0, name: str = 'MADDPG_DoD'):
        self.cfg = config; self.env = env; self.name = name
        self.device = torch.device('cpu')
        torch.manual_seed(seed); np.random.seed(seed)

        self.s_dim = config.get_state_dim()             # 54
        self.c_dim = config.get_critic_state_dim()      # 245
        self.a_dim = config.get_action_dim()            # 5

        self.actor        = _Actor(self.s_dim, self.a_dim).to(self.device)
        self.actor_target = _Actor(self.s_dim, self.a_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic        = _Critic(self.c_dim, self.a_dim).to(self.device)
        self.critic_target = _Critic(self.c_dim, self.a_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay = _ReplayBuffer(self.s_dim, self.c_dim, self.a_dim, replay_size)

        self.gamma = gamma; self.tau = tau
        self.tau_gs_start = tau_gs_start; self.tau_gs_end = tau_gs_end
        self.anneal_slots = anneal_slots; self.expl_noise = expl_noise
        self.batch_size = batch_size; self.start_steps = start_steps
        self.updates_per_slot = updates_per_slot; self.reward_scale = reward_scale
        # Zhong 忠实 reward (eq41) 状态
        self.zhong_reward = zhong_reward
        self.upsilon = float(upsilon) if upsilon is not None else float(getattr(config, 'UPSILON', 1.0))
        self.zw_q = float(zw_q); self.zw_y = float(zw_y)
        self._Y: Dict[int, float] = {}       # 时延虚拟队列 Y_n
        self._prevQ: Dict[int, float] = {}   # 上一槽队列(bytes)，算 l_n = Q_now - Q_prev

        self._eval_mode = False
        self._env_steps = 0
        self._cobs: Dict[int, np.ndarray] = {}          # 本槽各星 critic 态快照
        # 槽级缓冲：(s, cobs, a_onehot, mask, action_cost, sat_id)
        self._slot_tasks: List[Tuple] = []
        self._pending: Optional[Tuple] = None           # (s, cobs, a, r) 待补 s'/cobs'/mask'
        self.learning_curve: List[Dict] = []

    # ── Gumbel-Softmax ────────────────────────────────────────
    def _gs_temp(self) -> float:
        frac = min(self._env_steps / max(self.anneal_slots, 1), 1.0)
        return self.tau_gs_start + (self.tau_gs_end - self.tau_gs_start) * frac

    def _gumbel_softmax(self, logits, mask, tau, hard):
        logits = logits.masked_fill(mask < 0.5, -1e9)
        u = torch.rand_like(logits).clamp_min(1e-20)
        g = -torch.log((-torch.log(u)).clamp_min(1e-20))     # Gumbel(0,1)
        y = F.softmax((logits + g) / max(tau, 1e-6), dim=-1)
        if hard:
            idx = y.argmax(dim=-1, keepdim=True)
            y = (torch.zeros_like(y).scatter_(-1, idx, 1.0) - y).detach() + y
        return y

    # ── 槽级钩子：env.step 开头快照各星 critic 态 + 重置槽缓冲 ──
    def collect_critic_values(self, env: "SatelliteMECEnv") -> None:
        self._cobs = {n: np.asarray(v, np.float32) for n, v in env.get_critic_obs().items()}
        self._slot_tasks = []

    # ── 动作：局部态 → Gumbel-Softmax → 掩码 → 离散 ────────────
    def act_one(self, state: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        s_t = torch.as_tensor(np.asarray(state, np.float32), device=self.device).unsqueeze(0)
        m_t = torch.as_tensor(np.asarray(mask, np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s_t)
            if not self._eval_mode and self.expl_noise > 0:
                logits = logits + torch.randn_like(logits) * self.expl_noise
            if self._eval_mode:
                action = int(logits.masked_fill(m_t < 0.5, -1e9).argmax(dim=-1).item())
            else:
                action = int(self._gumbel_softmax(logits, m_t, self._gs_temp(), hard=True)
                             .argmax(dim=-1).item())
        return action, 0.0

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        """batch 兜底接口（env 检测到 act_one 走 sequential，不会用到此处）。"""
        self._eval_mode = True
        actions: Dict[int, List[int]] = {}
        for n in range(self.cfg.N_SATS):
            actions[n] = [self.act_one(np.asarray(s, np.float32), m)[0]
                          for s, m in zip(obs.get(n, []), masks.get(n, []))]
        return actions

    # ── 逐任务转移收集 ────────────────────────────────────────
    def record_task_transition(self, sat_id, slot_t, state, action,
                               log_prob, mask, task_reward) -> None:
        if self._eval_mode:
            return
        a_onehot = np.zeros(self.a_dim, np.float32); a_onehot[int(action)] = 1.0
        cobs = self._cobs.get(int(sat_id), np.zeros(self.c_dim, np.float32))
        self._slot_tasks.append((np.asarray(state, np.float32), cobs, a_onehot,
                                 np.asarray(mask, np.float32), float(task_reward), int(sat_id)))

    def _flush_slot(self, rewards: Dict, done: bool, info: Optional[Dict] = None) -> None:
        """把本槽任务串成每任务转移入回放。

        默认口径: r = action_cost + 分摊 outcome(rewards[n] − Σaction_cost, 按任务数均摊)。
        Zhong 口径(zhong_reward=True): 无视 env outcome/action_cost, 用忠实 eq41 的
        per-slot per-sat reward −(Q·l + Y·(T−Tmax) + υ·D) 分摊到该星本槽任务。
        """
        if not self._slot_tasks:
            return
        count: Dict[int, int] = {}
        for (_, _, _, _, _, n) in self._slot_tasks:
            count[n] = count.get(n, 0) + 1
        if self.zhong_reward and info is not None:
            r_sat = self._zhong_slot_reward(info, count)
            per_task = {n: r_sat[n] / max(count[n], 1) for n in count}
        else:
            sum_cost: Dict[int, float] = {}
            for (_, _, _, _, ac, n) in self._slot_tasks:
                sum_cost[n] = sum_cost.get(n, 0.0) + ac
            outcome = {n: float(rewards.get(n, 0.0)) - sum_cost[n] for n in count}
        for (s, cobs, a, mask, ac, n) in self._slot_tasks:
            if self.zhong_reward and info is not None:
                r = per_task[n] * self.reward_scale
            else:
                r = (ac + outcome[n] / max(count[n], 1)) * self.reward_scale
            if self._pending is not None:            # 上一条 pending 的 s'/cobs'/mask' = 当前任务
                ps, pc, pa, pr = self._pending
                self.replay.add(ps, pc, pa, pr, s, cobs, mask, 0.0)
                self._env_steps += 1
            self._pending = (s, cobs, a, r)
        self._slot_tasks = []
        if done and self._pending is not None:       # rollout 结束收尾（terminal）
            ps, pc, pa, pr = self._pending
            self.replay.add(ps, pc, pa, pr, np.zeros(self.s_dim, np.float32),
                            np.zeros(self.c_dim, np.float32), np.ones(self.a_dim, np.float32), 1.0)
            self._env_steps += 1
            self._pending = None

    def _zhong_slot_reward(self, info: Dict, count: Dict[int, int]) -> Dict[int, float]:
        """Zhong eq41 逐槽逐星 reward: r_n = −(zw_q·Q̂·l̂ + zw_y·Ŷ·excesŝ + υ·D̂)。

        映射(→ 我们的 env):
          Q_n  = qf_size+qb_size(bytes);   l_n = Q_now − Q_prev(逐槽变化, =arrival−departure)
          Y_n  = max(Y + (delay_mean − Tmax), 0)(时延虚拟队列, 仅本槽有完成时更新)
          D_n  = (comp+trans 能耗)/E_CAP(每槽放电流量, 即 Zhong 的 DoD)
        归一化: Q̂=Q/QUEUE_NORM, l̂=l/ZHONG_L_NORM, D̂=D/DELTA_DOD_MAX, 时延项按 Tmax 无量纲。
        吞吐由 −Q·l 队列漂移驱动(热点星积压大→强烈奖励卸载/清队), υ 单调控 DoD; 无 W_DONE。
        """
        cfg = self.cfg
        Q  = info.get('per_sat_queue', [])
        E  = info.get('per_sat_energy', [])
        Ds = info.get('per_sat_delay_sum', [])
        Dc = info.get('per_sat_delay_cnt', [])
        Tmax = float(getattr(cfg, 'T_MAX_DELAY', 6.0))
        QN   = float(cfg.QUEUE_NORM) if cfg.QUEUE_NORM else 1.0
        LN   = float(getattr(cfg, 'ZHONG_L_NORM', 1.0e8))
        DN   = float(cfg.DELTA_DOD_MAX) + 1e-12
        out: Dict[int, float] = {}
        for n in count:                                    # 只给本槽有动作的星
            q_now  = float(Q[n]) if n < len(Q) else 0.0
            q_prev = self._prevQ.get(n, q_now)
            l_n    = q_now - q_prev
            has_done = (n < len(Dc) and Dc[n] > 0)
            d_mean = (Ds[n] / Dc[n]) if has_done else 0.0
            excess = (d_mean - Tmax) if has_done else 0.0  # 无完成任务不更新 Y
            y_now  = max(self._Y.get(n, 0.0) + excess, 0.0)
            self._Y[n] = y_now
            D_n    = (float(E[n]) / cfg.E_CAP) if n < len(E) else 0.0
            q_drift = (q_prev / QN) * (l_n / LN)
            y_drift = (y_now / Tmax) * (excess / Tmax)
            dod_pen = D_n / DN
            out[n] = -(self.zw_q * q_drift + self.zw_y * y_drift + self.upsilon * dod_pen)
        for n in range(len(Q)):                            # 全星更新 prevQ, 供下槽算 l_n
            self._prevQ[n] = float(Q[n])
        return out

    def on_episode_end(self) -> None:
        pass

    # ── 训练驱动 ──────────────────────────────────────────────
    def run_step(self, env: "SatelliteMECEnv") -> Tuple[Dict, bool, Dict]:
        _, rewards, done, info = env.step(policy=self)
        if not self._eval_mode:
            self._flush_slot(rewards, done, info)
            for _ in range(self.updates_per_slot):
                self._update()
        return rewards, done, info

    def _update(self) -> None:
        if self.replay.size < max(self.batch_size, self.start_steps):
            return
        s, c, a, r, s2, c2, m2, d = self.replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            a2 = self._gumbel_softmax(self.actor_target(s2), m2, self._gs_temp(), hard=True)
            y = r + (1.0 - d) * self.gamma * self.critic_target(c2, a2)
        q = self.critic(c, a)
        critic_loss = F.smooth_l1_loss(q, y)
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5); self.critic_opt.step()

        # Actor：最大化中心化 Q（可微 Gumbel-Softmax 软动作，无掩码软 relax）
        a_soft = self._gumbel_softmax(self.actor(s), torch.ones_like(a), self._gs_temp(), hard=False)
        actor_loss = -self.critic(c, a_soft).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5); self.actor_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ── 评估 ──────────────────────────────────────────────────
    def quick_eval(self, env, n_slots: int = 800) -> Tuple[float, float, float]:
        cfg = self.cfg
        rng_task  = env.constellation.rng_task.bit_generator.state
        rng_param = env.constellation.rng_task_param.bit_generator.state

        self.set_eval_mode()
        env.reset(phase='eval', seeds=cfg.get_quick_eval_seeds())
        self._pending = None
        dod, hl = [], []
        for _ in range(n_slots):
            _, _, info = self.run_step(env)
            dod.append(info['avg_dod']); hl.append(info.get('avg_health_loss', 0.0))
        cr = env.get_eval_completion_rate()
        avg_dod, avg_hl = float(np.mean(dod)), float(np.mean(hl))

        self.set_train_mode()
        env.constellation.rng_task.bit_generator.state = rng_task
        env.constellation.rng_task_param.bit_generator.state = rng_param
        env.reset(phase='train'); self._pending = None
        self._Y.clear(); self._prevQ.clear()        # Zhong 队列跨 eval 间隔重置
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
        meta = dict(extra_info or {})
        with open(os.path.join(path, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
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
