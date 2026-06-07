"""
core/lyapunov.py
================
Lyapunov 代价函数计算器。

提供本地计算代价和转发代价的归一化计算，
以及 DoD（放电深度）增量的估算。

使用示例
--------
    from core.config import Config
    from core.lyapunov import LyapunovCalculator

    cfg  = Config()
    calc = LyapunovCalculator(cfg)

    # 获取卫星当前状态字典
    sat_state = calc.get_sat_state(satellite)

    # 计算本地代价
    cost = calc.normalized_local_cost(task, sat_state, nb_next=2, current_slot=100)

    # 计算转发代价
    nb_state = calc.get_neighbor_state(neighbor_info)
    cost = calc.normalized_forward_cost(task, sat_state, nb_state, b_nm=200e6, current_slot=100)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from core.config import Config
    from core.task import Task
    from core.satellite import Satellite


class LyapunovCalculator:
    """
    Lyapunov 代价函数计算器（无状态，线程安全）。

    Parameters
    ----------
    config          : Config 实例
    use_dod_penalty : 是否启用 DoD 虚拟队列惩罚项（默认True）
    use_battery_loss: 是否启用电池健康损失项（默认True）
    linear_loss     : 是否使用线性健康损失代替非线性（消融用，默认False）
    """

    def __init__(self, config: "Config",
                 use_dod_penalty: bool = True,
                 use_battery_loss: bool = True,
                 linear_loss: bool = False):
        self.cfg = config
        self.use_dod_penalty  = use_dod_penalty
        self.use_battery_loss = use_battery_loss
        self.linear_loss      = linear_loss

    # ── 电池健康损失函数 ──────────────────────────────────────
    def health_loss(self, dod: float) -> float:
        """L(δ) = δ · 10^{a(δ-1)}，linear_loss=True 时退化为 L(δ) = δ。"""
        if self.linear_loss:
            return dod
        a = self.cfg.A_COEF
        return dod * (10 ** (a * (dod - 1)))

    def health_loss_deriv(self, dod: float) -> float:
        """∂L/∂δ，用于一阶近似增量计算。"""
        if self.linear_loss:
            return 1.0
        a = self.cfg.A_COEF
        return (10 ** (a * (dod - 1))) * (1.0 + a * math.log(10) * dod)

    # ── DoD 增量估算 ──────────────────────────────────────────
    def delta_dod_comp(self, task: "Task", sat_state: Dict) -> float:
        """DVFS-aware 边际 DoD：接纳一个任务前后整星 slot 能耗的差分。

        与"独占 f_max 上界"不同，这里调用 select_freq 计算 f_before / f_after：
            f_before = sqrt(Q_hat / (3 V κ))  ∨ floor_hat
            f_after  = sqrt((Q_hat + S·H) / (3 V κ))  ∨ max(floor_hat, new_floor)
        然后 Δ E = κ · (f_after³ − f_before³) · τ。

        sat_state 必须含 q_cycles_hat、f_floor_hat、nb_hat（由 Satellite 在
        init_temp_state 与 apply_action 中维护）。
        """
        from core.dvfs import select_freq
        cfg = self.cfg

        q_before     = float(sat_state.get('q_cycles_hat', 0.0))
        floor_before = float(sat_state.get('f_floor_hat', 0.0))
        nb_next      = int(sat_state.get('nb_hat', 0)) + 1

        task_cycles    = task.size * task.cpu_cycles
        q_after        = q_before + task_cycles
        new_floor_task = nb_next * task_cycles / max(task.deadline, cfg.TAU)
        floor_after    = max(floor_before, new_floor_task)

        f_before = select_freq(cfg, q_before, f_floor=floor_before)
        f_after  = select_freq(cfg, q_after,  f_floor=floor_after)

        delta_e = cfg.KAPPA * cfg.TAU * max(f_after ** 3 - f_before ** 3, 0.0)
        return delta_e / cfg.E_CAP

    def delta_dod_trans(self, task: "Task", b_nm: float) -> float:
        """转发一个任务（链路速率 b_nm）产生的 DoD 增量。"""
        cfg = self.cfg
        if b_nm <= 0:
            return 0.0
        return cfg.P_T * (task.size / b_nm) / cfg.E_CAP

    def delta_health_loss_comp(self, task: "Task", sat_state: Dict, dod: float) -> float:
        """计算任务引起的健康损失增量（计算部分）。"""
        return self.health_loss_deriv(dod) * self.delta_dod_comp(task, sat_state)

    def delta_health_loss_trans(self, task: "Task", b_nm: float, dod: float) -> float:
        """计算任务引起的健康损失增量（传输部分）。"""
        return self.health_loss_deriv(dod) * self.delta_dod_trans(task, b_nm)

    # ── 归一化代价函数 ────────────────────────────────────────
    def normalized_local_cost(self, task: "Task", sat_state: Dict,
                               nb_next: int, current_slot: int) -> float:
        """
        本地计算的 Lyapunov 归一化代价。

        Parameters
        ----------
        task        : 待调度任务
        sat_state   : get_sat_state() 的返回值
        nb_next     : 接受该任务后的并发数（= nb_hat + 1）
        current_slot: 当前时隙

        Returns
        -------
        float : 归一化代价（无量纲）
        """
        cfg = self.cfg
        dod = sat_state['dod']; n_f = sat_state['n_f']
        nb_hat = sat_state['nb_hat']; z_hat = sat_state['z_hat']
        tilde_N_F = n_f - cfg.THETA_NUM
        remain = max(task.remain_time(current_slot), 1e-3)
        urgency = 1.0 - remain / cfg.D_MAX_MAX
        base_queue = task.size / (cfg.Q_NORM + 1e-9) / (1 + cfg.V)
        queue_item = base_queue * (nb_hat - tilde_N_F) + base_queue * urgency
        loss_item = 0.0
        if self.use_battery_loss:
            delta_l = self.delta_health_loss_comp(task, sat_state, dod)
            loss_item = (cfg.V / (1 + cfg.V)) * delta_l / (cfg.L_MAX_NEW_RAW + 1e-9)
        dod_item = 0.0
        if self.use_dod_penalty:
            delta_dod = self.delta_dod_comp(task, sat_state)
            dod_item = cfg.ETA * z_hat * delta_dod / (cfg.Z_MAX * cfg.DELTA_DOD_MAX + 1e-9)
        return queue_item + loss_item + dod_item

    def normalized_forward_cost(self, task: "Task", sat_state: Dict,
                                 neighbor_state: Dict, b_nm: float,
                                 current_slot: int) -> float:
        """
        转发的 Lyapunov 归一化代价。

        Parameters
        ----------
        task           : 待转发任务
        sat_state      : 本卫星的 get_sat_state() 返回值
        neighbor_state : get_neighbor_state() 返回值
        b_nm           : bits/s，链路速率
        current_slot   : 当前时隙

        Returns
        -------
        float : 归一化代价（无量纲）
        """
        cfg = self.cfg
        dod = sat_state['dod']; n_f_n = sat_state['n_f']
        n_f_m = neighbor_state.get('n_f', 0); z_hat = sat_state['z_hat']
        tilde_N_F_n = n_f_n - cfg.THETA_NUM
        tilde_N_F_m = n_f_m - cfg.THETA_NUM
        remain = max(task.remain_time(current_slot), 1e-3)
        urgency = 1.0 - remain / cfg.D_MAX_MAX
        base_queue = task.size / (cfg.Q_NORM + 1e-9) / (1 + cfg.V)
        queue_item = base_queue * (tilde_N_F_m - tilde_N_F_n) + base_queue * urgency
        loss_item = 0.0
        if self.use_battery_loss:
            delta_l = self.delta_health_loss_trans(task, b_nm, dod)
            loss_item = (cfg.V / (1 + cfg.V)) * delta_l / (cfg.L_MAX_NEW_RAW + 1e-9)
        dod_item = 0.0
        if self.use_dod_penalty:
            delta_dod = self.delta_dod_trans(task, b_nm)
            dod_item = cfg.ETA * z_hat * delta_dod / (cfg.Z_MAX * cfg.DELTA_DOD_MAX + 1e-9)
        return queue_item + loss_item + dod_item

    # ── 状态提取辅助 ──────────────────────────────────────────
    def get_sat_state(self, satellite: "Satellite") -> Dict:
        """从卫星对象提取用于代价计算的状态字典。

        q_cycles_hat / f_floor_hat 是本时隙累积的 DVFS 输入估计（含本时隙
        已接纳的新任务），用于 delta_dod_comp 做差分。
        """
        return {
            'dod':          satellite.dod,
            'n_f':          len(satellite.forward_queue),
            'qf_size':      satellite.qf_size,
            'nb_hat':       satellite.nb_hat,
            'z_hat':        satellite.z_hat,
            'q_cycles_hat': satellite.q_cycles_hat,
            'f_floor_hat':  satellite.f_floor_hat,
        }

    def get_neighbor_state(self, neighbor_info: Dict) -> Dict:
        """从邻居信息字典提取用于代价计算的状态字典。"""
        return {
            'qf_size': neighbor_info.get('qf_size', 0.0),
            'n_f':     neighbor_info.get('n_f', 0),
        }

    def compute_update_delta_dod_comp_timeslot(self, f_cmp: float) -> float:
        """单时隙计算功耗产生的 DoD 增量（Li-style DVFS：P=κf³, E=Pτ）。"""
        cfg = self.cfg
        if f_cmp <= 0:
            return 0.0
        return cfg.TAU * cfg.KAPPA * (f_cmp ** 3) / cfg.E_CAP

    def compute_update_delta_dod_trans_timeslot(
        self, forwarded_tasks: List[Tuple]
    ) -> float:
        """
        单时隙传输功耗产生的 DoD 增量。

        Parameters
        ----------
        forwarded_tasks : [(neighbor_id, task, b_nm), ...]
        """
        cfg = self.cfg
        total = sum(task.size / b_nm for _, task, b_nm in forwarded_tasks if b_nm > 0)
        return cfg.P_T * total / cfg.E_CAP
