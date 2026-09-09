"""
baselines/mhspo.py
==================
MHSPO 策略复现。

参考文献
--------
Zhang et al., "Energy-Efficient Computation Peer Offloading
in Satellite Edge Computing Networks", IEEE TMC 2024.

主要修正（相对于原始论文公式）
------------------------------
[A] DOGDPredictor 量纲：QB 用任务数（sat.nb），非 sat.qb_size(bits)
[B] _est_comp_delay_energy：QB 动态收缩模拟（Algorithm 1 循环语义）
[C] _forward_cost bracket 项：全部换算为任务数，消除 ~1e9 数值爆炸
[D] 代价函数整体 / Q_NORM 归一化，与 LyapunovCalculator 量级一致

使用示例
--------
    from baselines.mhspo import MHSPOPolicy

    policy = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=50.0)
    env.reset(phase='eval')
    for _ in range(5400):
        _, _, _, info = env.step(policy=policy)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from interfaces import PolicyInterface

if TYPE_CHECKING:
    from core.config import Config
    from core.env import SatelliteMECEnv
    from core.satellite import Satellite
    from core.task import Task


class DOGDPredictor:
    """
    Delayed Online Gradient Descent 工作负载预测器。

    对应论文 Algorithm 1 / 式(17)(18)。
    预测目标：QB_n*(tau)，单位【任务数】（对应 sat.nb）。

    Parameters
    ----------
    config  : Config 实例
    sat_id  : 所属卫星ID

    使用示例
    --------
        predictor = DOGDPredictor(cfg, sat_id=0)
        for t in range(5400):
            predictor.update(current_slot=t, actual_nb=sat.nb)
            predicted = predictor.get(tau=t+1, fallback=3.0)
    """

    def __init__(self, config: "Config", sat_id: int):
        self.cfg   = config
        self.sat_id = sat_id
        self.d_max  = int(config.D_MAX_MAX)
        self.qb_max = float(config.MAX_DISPATCH)
        self.T_elapsed      = 1
        self.predictions: Dict[int, float] = {}
        self._history_sum   = float(config.MAX_DISPATCH / 2)
        self._history_count = 1

    def _eta(self) -> float:
        """Algorithm 1 line 1: eta_n = QB_max_n / sqrt(T + D)"""
        D = 0.5 * self.T_elapsed * self.d_max * (1 + self.d_max)
        return self.qb_max / max(math.sqrt(self.T_elapsed + D), 1e-9)

    def _history_mean(self) -> float:
        return self._history_sum / max(self._history_count, 1)

    def update(self, current_slot: int, actual_nb: int) -> None:
        """
        观测 QB_n(t) = actual_nb 后，对未来窗口执行 DOGD 更新。

        Parameters
        ----------
        current_slot : 当前时隙
        actual_nb    : 实际并发任务数（sat.nb）
        """
        actual_f = float(actual_nb)
        self._history_sum   += actual_f
        self._history_count += 1
        self.qb_max  = max(self.qb_max, actual_f, 1.0)
        self.T_elapsed += 1
        eta       = self._eta()
        hist_mean = self._history_mean()
        pred_t    = self.predictions.get(current_slot, hist_mean)

        if   pred_t > actual_f + 1e-6: grad =  1.0
        elif pred_t < actual_f - 1e-6: grad = -1.0
        else:                           grad =  0.0

        for tau in range(current_slot + 1, current_slot + self.d_max + 1):
            old = self.predictions.get(tau, hist_mean)
            self.predictions[tau] = float(np.clip(old - eta * grad, 0.0, self.qb_max))

        # 清理过期预测
        for k in [k for k in list(self.predictions) if k < current_slot]:
            del self.predictions[k]

    def get(self, tau: int, fallback: float) -> float:
        """
        获取时隙 tau 的预测值。

        Parameters
        ----------
        tau      : 目标时隙
        fallback : 无预测时的默认值

        Returns
        -------
        float : 预测并发任务数（≥ 0）
        """
        return self.predictions.get(tau, max(fallback, 0.0))


class MHSPOPolicy(PolicyInterface):
    """
    Multi-Hop Satellite Peer Offloading（MHSPO）策略。

    Parameters
    ----------
    config      : Config 实例
    env         : SatelliteMECEnv 实例
    rho_d       : 时延权重（默认1.0）
    rho_e       : 能耗权重（默认1.0）
    V_lyapunov  : Lyapunov 权衡参数（默认使用 config.V）
    name        : 策略名称
    """

    needs_training: bool = False
    name:           str  = 'MHSPO'

    def __init__(self, config: "Config", env: "SatelliteMECEnv",
                 rho_d: float = 1.0, rho_e: float = 1.0,
                 V_lyapunov: Optional[float] = None,
                 name: str = 'MHSPO'):
        self.cfg        = config
        self.env        = env
        self.rho_d      = rho_d
        self.rho_e      = rho_e
        self.V_lyapunov = V_lyapunov if V_lyapunov is not None else config.V
        self.name       = name

        cfg = config
        self.B_bits_max:   float = cfg.B_MAX * cfg.TAU
        self.theta_n_bits: float = 2.0 * (
            cfg.S_MAX * cfg.MAX_DISPATCH + (cfg.N_NEIGHBORS - 1) * self.B_bits_max
        )
        self.Q_NORM: float = cfg.Q_NORM

        self.predictors: Dict[int, DOGDPredictor] = {
            n: DOGDPredictor(config, n) for n in range(config.N_SATS)
        }
        self._link_used:   Dict[int, Dict[int, float]] = {}
        self._local_count: Dict[int, int]              = {}

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        t           = self.env.current_slot
        global_info = self.env._global_info
        sats        = self.env.constellation.satellites

        # Algorithm 1：DOGD 更新
        for sat in sats:
            self.predictors[sat.sat_id].update(t, sat.nb)

        for n in range(self.cfg.N_SATS):
            self._link_used[n]   = {}
            self._local_count[n] = 0

        result = {sat.sat_id: self._greedy_schedule(sat, t, global_info) for sat in sats}
        # The environment mask is the authoritative final feasibility gate.
        for n, actions in result.items():
            for i, mask in enumerate(masks.get(n, [])):
                if mask.sum() and not mask[actions[i]]:
                    actions[i] = int(np.argmax(mask))
        return result

    def _greedy_schedule(self, sat: "Satellite", t: int,
                         global_info: Dict) -> List[int]:
        """P3 式(24) 贪心：枚举所有可行动作，选最小归一化代价。"""
        cfg   = self.cfg
        n     = sat.sat_id
        tasks = list(sat.forward_queue)
        if not tasks:
            return []

        qf_bits   = sat.qf_size
        qb_num    = float(sat.nb)
        q_tilde_f = qf_bits - self.theta_n_bits

        window  = int(cfg.D_MAX_MAX) + 2
        qb_pred = {
            tau: self.predictors[n].get(tau, qb_num)
            for tau in range(t, t + window)
        }

        sat_actions: List[int] = []
        for task in tasks:
            best_action = 0
            best_cost   = float('inf')

            # 滚动并发数：包含本时隙已分配到本地的任务，避免过度乐观
            qb_num_eff = qb_num + float(self._local_count[n])

            # 动作0：本地计算（约束 12c）
            if self._local_count[n] < cfg.MAX_DISPATCH:
                d_comp, e_comp = self._est_comp_delay_energy(
                    task, t + 1, qb_num_eff, qb_pred, cfg.CPU_FREQ)
                if d_comp <= task.remain_time(t) + cfg.TAU:
                    c = self._local_cost_norm(task, qb_num_eff, q_tilde_f, d_comp, e_comp)
                    if c < best_cost:
                        best_cost, best_action = c, 0

            # 动作1~N：转发（约束 12b、12d）
            for idx, nb_id in enumerate(sat.neighbors):
                b_nm        = sat.link_rates.get(nb_id, cfg.B_AVG)
                t_nm        = sat.prop_delays.get(nb_id, 0.0)
                nb_nb_count = global_info.get(nb_id, {}).get('nb', 0)
                if not task.feasible_forward(b_nm, t_nm, nb_nb_count,
                                             cfg.CPU_FREQ, cfg.TAU, t, cfg.K_MAX):
                    continue
                if self._link_used[n].get(nb_id, 0.0) + task.size > self.B_bits_max:
                    continue
                # 邻居感知：用 DOGD 预测器读取邻居 QB 在任务到达时的预测值
                arrival_slot = t + int(math.ceil((task.size / b_nm + t_nm) / cfg.TAU))
                nb_qb_pred   = self.predictors[nb_id].get(arrival_slot, float(nb_nb_count))
                c = self._forward_cost_norm(task, b_nm, t_nm, qf_bits, q_tilde_f,
                                            nb_qb_pred)
                if c < best_cost:
                    best_cost, best_action = c, idx + 1

            # 更新容量计数器
            if best_action == 0:
                self._local_count[n] += 1
            else:
                nb_id = sat.neighbors[best_action - 1]
                self._link_used[n][nb_id] = self._link_used[n].get(nb_id, 0.0) + task.size

            sat_actions.append(best_action)
        return sat_actions

    def _est_comp_delay_energy(self, task: "Task", start_slot: int,
                               qb_num_now: float, qb_pred: Dict[int, float],
                               cpu_freq: float) -> Tuple[float, float]:
        """
        Algorithm 1 lines 10-15：估算 d^{i,C}_n 和 e^{i,C}_n。

        [FIX-B] nb_running 动态收缩：模拟任务完成后队列缩减。
        """
        cfg       = self.cfg
        remaining = task.size
        energy    = 0.0
        tau       = start_slot
        max_tau   = start_slot + int(cfg.D_MAX_MAX) + 2
        nb_running = max(qb_pred.get(start_slot, qb_num_now) + 1.0, 1.0)

        while remaining > 1e-9 and tau < max_tau:
            processed = min(cpu_freq * cfg.TAU / task.cpu_cycles, remaining)
            freq_e    = cpu_freq / nb_running   # frequency-division energy model
            energy   += cfg.KAPPA * processed * task.cpu_cycles * (freq_e ** 2)
            remaining -= processed
            tau       += 1
            nb_running = max(qb_pred.get(tau, max(nb_running - 1.0, 1.0)), 1.0)

        return float((tau - start_slot) * cfg.TAU), energy

    def _local_cost_norm(self, task: "Task", qb_num: float,
                         q_tilde_f: float, d_comp: float, e_comp: float) -> float:
        """P3 第2项归一化：[FIX-C/D] 全部换算为任务数 + / Q_NORM 归一化。"""
        cfg           = self.cfg
        s_avg         = cfg.S_AVG
        s_i_num       = task.size / s_avg
        q_tilde_f_num = q_tilde_f / s_avg
        raw = (s_i_num * qb_num
               - s_i_num * q_tilde_f_num
               + self.V_lyapunov * (self.rho_d * d_comp + self.rho_e * e_comp))
        return raw / (self.Q_NORM + 1e-9)

    def _forward_cost_norm(self, task: "Task", b_nm: float, t_nm: float,
                           qf_bits: float, q_tilde_f: float,
                           nb_qb_pred: float) -> float:
        """
        P3 第3项归一化：[FIX-A/D] bracket 换算为任务数 + / Q_NORM 归一化。

        nb_qb_pred 邻居预测 QB（任务数），由 DOGD 预测器提供。
        加入 +s_i_num * nb_qb_pred 惩罚项，使 MHSPO 主动避开高负载邻居。
        """
        cfg   = self.cfg
        s_avg = cfg.S_AVG

        trans_delay  = task.size / b_nm + t_nm
        trans_energy = (task.size / b_nm) * cfg.P_T

        s_i_num       = task.size / s_avg
        s_max_num     = cfg.S_MAX / s_avg
        q_tilde_f_num = q_tilde_f / s_avg
        bracket_num   = min(qf_bits - self.theta_n_bits, 0.0) / s_avg

        raw = (self.V_lyapunov * self.rho_d * trans_delay
               + self.V_lyapunov * self.rho_e * trans_energy
               - s_i_num   * q_tilde_f_num
               - s_max_num * bracket_num
               + s_i_num   * nb_qb_pred)
        return raw / (self.Q_NORM + 1e-9)
