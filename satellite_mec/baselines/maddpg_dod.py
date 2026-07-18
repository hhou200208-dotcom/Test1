"""
baselines/maddpg_dod.py
=======================
LyDRL-DoD —— 基于 MADDPG 的 Lyapunov-DoD 卸载基线（论文忠实复现）。

论文复现（V2 忠实口径，与爸爸讨论后定稿）
------------------------------------------
Liang Zhong, Shen Tian, Deze Zeng, Zhihao Qu, Chengyu Hu,
"Battery Lifetime Extension in Heterogeneous Satellite Edge Computing:
 A Lyapunov-DRL Approach,"
IEEE Internet of Things Journal, vol. 13, no. 8, pp. 15976-15988, Apr. 2026.
DOI: 10.1109/JIOT.2026.3658536

用途：作为 baseline 与本仓库 LyaMAPPO 头对头对比。原文方法 = Lyapunov
drift-plus-penalty + MADDPG + Gumbel-Softmax，目标是最小化长期平均 DoD 同时
保证队列/时延稳定。本文件**忠实复现原文的 MADDPG 学习器与奖励结构**，仅共享
本仓库的 Lyapunov-DVFS 物理基底以保证与其它策略的对比公平。

忠实复现的内容（对齐原文）
--------------------------
| 原文                                    | 本文件落地 |
|-----------------------------------------|-----------|
| 观测 o_{n,t}={Qₙ, Yₙ, Dₙ, E^harvestₙ}(式39) | 4 维，从 env 54 维态提取 Qₙ/Dₙ/E_harvest + 内部维护 Yₙ |
| 动作 x_{n,m}∈{local, 转发1邻居}(式40)   | 离散 {0=本地, 1..4=邻居}，dim=5（与 env 动作空间天然一致）|
| 奖励 r=−[ΣQₙ·lₙ + ΣYₙ·(T−T_max) + υ·ΣDₙ](式41) | 团队共享奖励，逐槽从 env 读回计算（见 _paper_team_reward）|
| Gumbel-Softmax 重参数化（式；Jang 2016）| Actor 输出 logits→掩码→Gumbel-Softmax 采样，温度退火 |
| MADDPG CTDE：中心化 critic + 分布式 actor | concat 中心化 critic（拼接所有 agent 的 obs+action，式：ΣN(dim o+dim a)=25×(4+5)=225）|
| Actor/Critic 2×256 ReLU, LR 1e-3, σ=0.01, Adam(Table II) | 全对齐 |
| Gaussian 探索 N(0,1)                    | 训练期给 logits 加噪 + Gumbel 采样；eval 期掩码 argmax |

适配说明（如实标注，遵循 td3_sched.py 先例）
--------------------------------------------
1. 原文每星每槽 1 个卸载决策；本仓库 env 每星每槽 forward_queue 有多任务，
   逐任务决策。采用**每星每槽同一 4 维状态**对每个任务决策（原文观测本就是
   星级/槽级、非任务级），符合原文语义；转移在**槽级**聚合成 MADDPG 联合转移。
2. concat critic 的"联合动作"由每星本槽任务动作的归一化 one-hot 剖面（action
   profile）表示；无任务的星取 [1,0,0,0,0]（本地/空操作）。
3. Yₙ（时延虚拟队列，式21）：原文为每星维护；本文件用**系统级** Y_sys 维护
   （奖励本为团队求和 ΣYₙ·(·)，系统级聚合等价且非侵入 env）。
4. Dₙ（式14-15，本槽放电分数 = max(能耗−采集,0)/E_CAP）从 env 逐槽能耗字段
   (slot_comp/trans/house_energy, solar_power) 重构，与原文一致。
5. 队列漂移项 ΣQₙ·lₙ 与 DoD 惩罚项 υ·ΣDₙ 分别归一化到 O(1) 量级；υ 为原文
   Lyapunov 控制参数（Fig.11 扫描量），此处可调，冒烟测试后校准。

接口与 TD3SchedPolicy/MAPPOPolicy 对齐（act_one / record_task_transition /
run_step / quick_eval / save / load），可直接复用 ExperimentRunner 与 eval_*。
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
    """论文式 Actor：obs(4) → 256 → 256 → logits(5)。掩码 + Gumbel-Softmax 在外部施加。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, action_dim)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(o))
        x = F.relu(self.l2(x))
        return self.l3(x)                      # logits（未 softmax）


class _Critic(nn.Module):
    """中心化 concat critic：joint(obs+action) → 256 → 256 → Q(1)。

    输入维度 = N·(obs_dim + action_dim)（原文中心化 critic 结构，随 N 线性增长）。
    """

    def __init__(self, joint_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(joint_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, joint_obs: torch.Tensor, joint_act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_obs, joint_act], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# ──────────────────────────────────────────────────────────────
# 回放池（槽级 MADDPG 联合转移）
# ──────────────────────────────────────────────────────────────
class _JointReplay:
    """存储槽级联合转移 (joint_obs, joint_act, r, joint_obs', done, joint_mask')。"""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 200_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.o  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r  = np.zeros((capacity, 1),       dtype=np.float32)
        self.o2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d  = np.zeros((capacity, 1),       dtype=np.float32)
        self.m2 = np.zeros((capacity, act_dim), dtype=np.float32)   # next-obs 联合掩码

    def add(self, o, a, r, o2, done, m2):
        i = self.ptr
        self.o[i] = o; self.a[i] = a; self.r[i] = r
        self.o2[i] = o2; self.d[i] = done; self.m2[i] = m2
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device):
        idx = np.random.randint(0, self.size, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.o), to(self.a), to(self.r), to(self.o2), to(self.d), to(self.m2)


# ──────────────────────────────────────────────────────────────
# 策略
# ──────────────────────────────────────────────────────────────
class MADDPGDoDPolicy(PolicyInterface):
    """LyDRL-DoD：Lyapunov-DoD + MADDPG + Gumbel-Softmax（Zhong et al. IoT-J 2026 复现）。

    Parameters
    ----------
    config, env : 标准
    upsilon     : 原文 Lyapunov 控制参数 υ（DoD 惩罚权重，Fig.11 扫描量）
    w_queue, w_delay : 队列漂移项 / 时延虚拟队列项的归一化权重（适配 env 量纲）
    actor_lr, critic_lr : 学习率（Table II：均 1e-3）
    gamma, tau  : 折扣 / 目标网络软更新（Table II：σ=0.01）
    tau_gs_start, tau_gs_end : Gumbel-Softmax 温度退火起止
    expl_noise  : 训练期 logits 高斯噪声（对应原文 N(0,1) 探索）
    batch_size, start_steps, updates_per_slot, replay_size, reward_scale
    """

    needs_training: bool = True
    name: str = 'MADDPG_DoD'

    # 54 维 actor 态中，论文 4 维 obs 的抽取索引（见 satellite.get_state layout）
    _IDX_QF, _IDX_QB, _IDX_DOD, _IDX_SOLAR = 1, 2, 4, 9

    def __init__(self, config: "Config", env: "SatelliteMECEnv",
                 upsilon: float = 50.0, w_queue: float = 15.0, w_delay: float = 0.02,
                 t_max_delay: float = 3.0, y_max: float = 10.0,
                 actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01,
                 tau_gs_start: float = 1.0, tau_gs_end: float = 0.5,
                 anneal_slots: int = 30000, expl_noise: float = 0.3,
                 batch_size: int = 256, start_steps: int = 2000,
                 updates_per_slot: int = 1, replay_size: int = 200_000,
                 reward_scale: float = 0.02, cost_clip: float = 50.0,
                 seed: int = 0, name: str = 'MADDPG_DoD'):
        self.cfg = config
        self.env = env
        self.name = name
        self.device = torch.device('cpu')
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.N = config.N_SATS
        self.obs_dim_n = 4                              # 论文 4 维 obs
        self.act_dim_n = config.get_action_dim()        # 5
        self.joint_obs_dim = self.N * self.obs_dim_n    # 100
        self.joint_act_dim = self.N * self.act_dim_n    # 125

        # 网络：Actor 参数共享（团队共享奖励 + 结构同构 ⇒ 与逐 agent 网期望等价）
        self.actor        = _Actor(self.obs_dim_n, self.act_dim_n).to(self.device)
        self.actor_target = _Actor(self.obs_dim_n, self.act_dim_n).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic        = _Critic(self.joint_obs_dim + self.joint_act_dim).to(self.device)
        self.critic_target = _Critic(self.joint_obs_dim + self.joint_act_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = _JointReplay(self.joint_obs_dim, self.joint_act_dim, replay_size)

        # 超参
        self.upsilon = upsilon; self.w_queue = w_queue; self.w_delay = w_delay
        self.t_max_delay = t_max_delay          # 论文式 T_max：固定时延约束(Fig.9 为 1-5s)
        self.y_max = y_max                       # 时延虚拟队列上限（防未收敛期无界累积→惩罚爆炸）
        self.gamma = gamma; self.tau = tau
        self.tau_gs_start = tau_gs_start; self.tau_gs_end = tau_gs_end
        self.anneal_slots = anneal_slots; self.expl_noise = expl_noise
        self.batch_size = batch_size; self.start_steps = start_steps
        self.updates_per_slot = updates_per_slot; self.reward_scale = reward_scale
        self.cost_clip = cost_clip

        self._eval_mode = False
        self._env_steps = 0

        # 逐槽状态
        self._Y_sys: float = 0.0                        # 系统级时延虚拟队列（式21）
        self._prev_Q: np.ndarray = np.zeros(self.N, np.float32)   # 上一槽末各星总积压(bits)
        self._slot_act_count: np.ndarray = np.zeros((self.N, self.act_dim_n), np.float32)
        self._slot_mask_union: np.ndarray = np.zeros((self.N, self.act_dim_n), np.float32)
        self._joint_obs_start: Optional[np.ndarray] = None
        self._pending: Optional[Tuple] = None           # (joint_o, joint_a, r) 待补 s'
        # act_one → record_task_transition 的 sat 定位：按 env 顺序推进的游标
        self._cursor_sat: int = 0
        self._cursor_task_in_sat: int = 0

        self.learning_curve: List[Dict] = []

    # ── 论文 4 维 obs 抽取 ────────────────────────────────────
    def _paper_obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """从 env 54 维 actor 态提取论文式(39) 4 维 obs = [Qₙ, Yₙ, Dₙ, E^harvestₙ]。"""
        Q = float(state[self._IDX_QF] + state[self._IDX_QB])   # (qf+qb)/Q_F_MAX
        D = float(state[self._IDX_DOD])                        # dod/DOD_MAX
        H = float(state[self._IDX_SOLAR])                      # solar/P_SOLAR_MAX
        Y = min(self._Y_sys / max(self.cfg.D_MAX_MAX, 1e-6), 10.0) / 10.0
        return np.array([Q, Y, D, H], dtype=np.float32)

    def _sat_paper_obs(self, sat) -> np.ndarray:
        """直接从卫星对象构造论文 4 维 obs（用于槽级联合态，collect_critic_values 调用）。"""
        cfg = self.cfg
        Q = (sat.qf_size + sat.qb_size) / (cfg.Q_F_MAX + 1e-9)
        D = sat.dod / cfg.DOD_MAX
        H = sat.solar_power / max(cfg.P_SOLAR_MAX, 1e-6)
        Y = min(self._Y_sys / max(cfg.D_MAX_MAX, 1e-6), 10.0) / 10.0
        return np.array([Q, Y, D, H], dtype=np.float32)

    # ── Gumbel-Softmax（Jang et al. 2016）──────────────────────
    def _gs_temp(self) -> float:
        frac = min(self._env_steps / max(self.anneal_slots, 1), 1.0)
        return self.tau_gs_start + (self.tau_gs_end - self.tau_gs_start) * frac

    def _gumbel_softmax(self, logits: torch.Tensor, mask: torch.Tensor,
                        tau: float, hard: bool) -> torch.Tensor:
        """掩码 + Gumbel-Softmax。logits/mask: (B, A)。返回 (B, A) 软/硬 one-hot。"""
        logits = logits.masked_fill(mask < 0.5, -1e9)
        # Gumbel(0,1) = -log(-log(U))；注意括号：先取 (-log U)>0 再 clamp，避免对负数取 log
        u = torch.rand_like(logits).clamp_min(1e-20)
        g = -torch.log((-torch.log(u)).clamp_min(1e-20))
        y = F.softmax((logits + g) / max(tau, 1e-6), dim=-1)
        if hard:
            idx = y.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
            y = (y_hard - y).detach() + y            # straight-through
        return y

    # ── 槽级钩子：env.step 开头调用，快照联合态 + 重置槽累加器 ──
    def collect_critic_values(self, env: "SatelliteMECEnv") -> None:
        sats = env.constellation.satellites
        self._joint_obs_start = np.concatenate(
            [self._sat_paper_obs(sat) for sat in sats]).astype(np.float32)
        self._slot_act_count[:] = 0.0
        self._slot_mask_union[:] = 0.0
        self._cursor_sat = 0
        self._cursor_task_in_sat = 0

    # ── 动作：论文 obs → Gumbel-Softmax → 掩码 → 离散 ──────────
    def act_one(self, state: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
        o = self._paper_obs_from_state(np.asarray(state, np.float32))
        o_t = torch.as_tensor(o, device=self.device).unsqueeze(0)
        m_t = torch.as_tensor(np.asarray(mask, np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(o_t)
            if not self._eval_mode and self.expl_noise > 0:
                logits = logits + torch.randn_like(logits) * self.expl_noise
            if self._eval_mode:
                masked = logits.masked_fill(m_t < 0.5, -1e9)
                action = int(masked.argmax(dim=-1).item())
            else:
                y = self._gumbel_softmax(logits, m_t, self._gs_temp(), hard=True)
                action = int(y.argmax(dim=-1).item())
        return action, 0.0

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        """batch 兜底接口（env 检测到 act_one 走 sequential，不会用到此处）。"""
        actions: Dict[int, List[int]] = {}
        self._eval_mode = True
        for n in range(self.N):
            sat_actions = []
            for state, mask in zip(obs.get(n, []), masks.get(n, [])):
                a, _ = self.act_one(np.asarray(state, np.float32), mask)
                sat_actions.append(a)
            actions[n] = sat_actions
        return actions

    # ── 逐任务转移收集：累加每星 action profile + mask union ──
    def record_task_transition(self, sat_id, slot_t, state, action,
                               log_prob, mask, task_reward) -> None:
        if self._eval_mode:
            return
        n = int(sat_id)
        self._slot_act_count[n, int(action)] += 1.0
        self._slot_mask_union[n] = np.maximum(self._slot_mask_union[n],
                                              np.asarray(mask, np.float32))

    # ── 论文式(41) 团队奖励：逐槽从 env 读回计算 ───────────────
    def _paper_team_reward(self, env: "SatelliteMECEnv", info: Dict) -> float:
        cfg = self.cfg
        sats = env.constellation.satellites
        Q_now = np.array([(s.qf_size + s.qb_size) for s in sats], dtype=np.float64)

        # (a) 队列漂移 Σ Qₙ(t)·lₙ(t)，lₙ = ΔQ = 本槽末 − 上槽末（=w−b 净变化）
        l_n = Q_now - self._prev_Q
        qn = self._prev_Q / (cfg.Q_F_MAX + 1e-9)
        ln = l_n / (cfg.Q_F_MAX + 1e-9)
        queue_term = float(np.sum(qn * ln))

        # (b) 时延虚拟队列 Y_sys·(T̄−T_max)，系统级（式21 更新）
        #     T_max 用论文式固定时延约束（非 per-task deadline），使约束真正生效
        delays = info.get('slot_e2e_delays', []) or []
        Tbar  = float(np.mean(delays)) if delays else 0.0
        T_max = self.t_max_delay
        viol = Tbar - T_max
        self._Y_sys = min(max(self._Y_sys + viol, 0.0), self.y_max)   # 有界虚拟队列
        delay_term = self._Y_sys * viol / max(T_max, 1e-6)

        # (c) DoD 惩罚 υ·Σ Dₙ，Dₙ=max(能耗−采集,0)/E_CAP（式14-15）逐槽重构
        dod_term = 0.0
        for s in sats:
            e_consume = s.slot_comp_energy + s.slot_trans_energy + s.slot_house_energy
            e_harvest = s.solar_power * cfg.TAU
            dod_term += max(e_consume - e_harvest, 0.0) / cfg.E_CAP
        dod_term = self.upsilon * dod_term

        self._prev_Q = Q_now.astype(np.float32)
        cost = self.w_queue * queue_term + self.w_delay * delay_term + dod_term
        # 稳定性：裁剪罕见尖峰后缩放，避免 critic target 量级过大导致 DDPG 发散
        cost = float(np.clip(cost, -self.cost_clip, self.cost_clip))
        return -cost * self.reward_scale

    # ── 训练驱动（与 MAPPOPolicy.run_step / TD3 同签名）────────
    def run_step(self, env: "SatelliteMECEnv") -> Tuple[Dict, bool, Dict]:
        _, rewards, done, info = env.step(policy=self)
        if not self._eval_mode:
            self._flush_slot(env, info, done)
            for _ in range(self.updates_per_slot):
                self._update()
        return rewards, done, info

    def _flush_slot(self, env, info, done: bool) -> None:
        if self._joint_obs_start is None:
            return
        r = self._paper_team_reward(env, info)

        # 本槽每星 action profile（归一化 one-hot 剖面）；无任务星 → 本地 [1,0,0,0,0]
        prof = self._slot_act_count.copy()
        counts = prof.sum(axis=1, keepdims=True)
        no_task = (counts.squeeze(1) < 0.5)
        prof[~no_task] /= counts[~no_task]
        prof[no_task] = 0.0; prof[no_task, 0] = 1.0
        joint_a = prof.reshape(-1).astype(np.float32)

        # 本槽联合掩码（合法动作，作为上一条 pending 转移的 mask'，对应其 s'=本槽起始态）
        m = self._slot_mask_union.copy()
        m[no_task] = 0.0; m[no_task, 0] = 1.0
        joint_m = m.reshape(-1).astype(np.float32)

        # 补齐上一条 pending 的 s' 与 mask'：s'=本槽起始联合态，mask'=本槽联合掩码
        if self._pending is not None:
            po, pa, pr = self._pending
            self.replay.add(po, pa, pr, self._joint_obs_start, 0.0, joint_m)
            self._env_steps += 1
        # 本槽转移入 pending（s'/mask' 待下一槽补齐）
        self._pending = (self._joint_obs_start, joint_a, r)

        if done and self._pending is not None:
            po, pa, pr = self._pending
            sats = env.constellation.satellites
            joint_o2 = np.concatenate(
                [self._sat_paper_obs(s) for s in sats]).astype(np.float32)
            self.replay.add(po, pa, pr, joint_o2, 1.0, joint_m)   # done=1，s'/mask' 不参与 target
            self._env_steps += 1
            self._pending = None

    def on_episode_end(self) -> None:
        pass

    # ── MADDPG 更新 ───────────────────────────────────────────
    def _joint_target_action(self, o2: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """用 target actor 对每个 agent 的 next-obs 算 Gumbel-Softmax 联合动作（向量化）。"""
        B, N, ao = o2.shape[0], self.N, self.act_dim_n
        logits = self.actor_target(o2.view(B * N, self.obs_dim_n))
        a2 = self._gumbel_softmax(logits, m2.view(B * N, ao), self._gs_temp(), hard=True)
        return a2.view(B, N * ao)

    def _update(self) -> None:
        if self.replay.size < max(self.batch_size, self.start_steps):
            return
        o, a, r, o2, d, m2 = self.replay.sample(self.batch_size, self.device)

        # Critic 更新
        with torch.no_grad():
            a2 = self._joint_target_action(o2, m2)
            q_t = self.critic_target(o2, a2)
            y = r + (1.0 - d) * self.gamma * q_t
        q = self.critic(o, a)
        critic_loss = F.smooth_l1_loss(q, y)      # Huber：对大 TD 误差鲁棒，抑制发散
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        # Actor 更新：共享 actor，对每个 agent slot 替换其动作后最大化 Q。
        # 向量化：把 N 个"替换单 agent 动作"的联合动作堆成 (B·N) 批，单次 critic 前/反向。
        B = o.shape[0]; N, ao = self.N, self.act_dim_n
        on = o.view(B, N, self.obs_dim_n)
        a_cur = a.view(B, N, ao)
        tau = self._gs_temp()
        # 全 agent 当前 actor 的可微 Gumbel-Softmax 动作（无掩码软 relax）
        logits_all = self.actor(on.reshape(B * N, self.obs_dim_n))
        a_gs = self._gumbel_softmax(
            logits_all, torch.ones_like(logits_all), tau, hard=False).view(B, N, ao)
        # a_exp[b,n] = a_cur[b] 但第 n 个 agent 的动作替换为 a_gs[b,n]
        a_exp = a_cur.unsqueeze(1).expand(B, N, N, ao).clone()
        diag = torch.arange(N)
        a_exp[:, diag, diag, :] = a_gs
        o_exp = o.unsqueeze(1).expand(B, N, self.joint_obs_dim).reshape(B * N, -1)
        q_actor = self.critic(o_exp, a_exp.reshape(B * N, N * ao))
        actor_loss = -q_actor.mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        # 软更新 target（Table II：σ=0.01）
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ── 评估 ──────────────────────────────────────────────────
    def quick_eval(self, env, n_slots: int = 1000) -> Tuple[float, float, float]:
        cfg = self.cfg
        rng_task  = env.constellation.rng_task.bit_generator.state
        rng_param = env.constellation.rng_task_param.bit_generator.state

        self.set_eval_mode()
        env.reset(phase='eval', seeds=cfg.get_quick_eval_seeds())
        self._Y_sys = 0.0; self._prev_Q[:] = 0.0; self._pending = None
        dod, hl = [], []
        for _ in range(n_slots):
            _, _, info = self.run_step(env)
            dod.append(info['avg_dod']); hl.append(info.get('avg_health_loss', 0.0))
        cr = env.get_eval_completion_rate()
        avg_dod, avg_hl = float(np.mean(dod)), float(np.mean(hl))

        self.set_train_mode()
        env.constellation.rng_task.bit_generator.state = rng_task
        env.constellation.rng_task_param.bit_generator.state = rng_param
        env.reset(phase='train')
        self._Y_sys = 0.0; self._prev_Q[:] = 0.0; self._pending = None
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
        meta = {'upsilon': self.upsilon, 'w_queue': self.w_queue, 'w_delay': self.w_delay}
        if extra_info:
            meta.update(extra_info)
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
