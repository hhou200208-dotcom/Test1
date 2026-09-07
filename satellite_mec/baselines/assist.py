"""
baselines/assist.py
===================
assist 策略 —— 在 MHSPO 基础上，把代价中的**能耗项**替换为**电池寿命损耗项**。

动机
----
MHSPO(Zhang TMC2024)代价含能耗项 rho_e·E（本地 e_comp / 转发 trans_energy）。
本策略保留 MHSPO 的多跳贪心 + DOGD 邻居负载预测**完全不变**，仅将能耗 E 替换为
DoD 相关的边际寿命损耗代理：

    battery_term = L'(DoD_n) · (E / E_cap) / L_max_new      （归一化边际 HL，与 BLA-MAPPO 同构）

其中 L'(δ)=10^{a(δ-1)}(1+a·ln10·δ)（config.A_COEF）。这样调度在能量使用上被赋予
**状态相关**权重：高 DoD 卫星上的计算/传输被加重惩罚，从而降低累计寿命损耗。
rho_batt(原 rho_e 位)为电池权重旋钮：越大越回避高 DoD 放电、HL 越低。

说明
----
- 与 MHSPO 唯一区别是代价里的能耗→电池损耗替换；其余（DOGD、可行性、容量约束、
  归一化、贪心结构）完全继承，保证公平可比。
- 启发式，不需要离线训练；原 MHSPOPolicy 不受影响（本类为独立子类）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

from baselines.mhspo import MHSPOPolicy

if TYPE_CHECKING:
    from core.config import Config
    from core.env import SatelliteMECEnv
    from core.satellite import Satellite
    from core.task import Task


class AssistPolicy(MHSPOPolicy):
    """MHSPO 变体：能耗项 → 电池寿命损耗项。rho_batt 为电池权重旋钮。"""

    name: str = 'assist'

    def __init__(self, config: "Config", env: "SatelliteMECEnv",
                 rho_d: float = 1.0, rho_batt: float = 1.0,
                 V_lyapunov: Optional[float] = None, name: str = 'assist'):
        super().__init__(config, env, rho_d=rho_d, rho_e=rho_batt,
                         V_lyapunov=V_lyapunov, name=name)
        self._a = config.A_COEF
        self._ln10 = math.log(10.0)
        self._cur_dod = 0.0

    def _lprime(self, dod: float) -> float:
        """L'(DoD) = 10^{a(DoD-1)} (1 + a ln10 DoD)。"""
        return (10.0 ** (self._a * (dod - 1.0))) * (1.0 + self._a * self._ln10 * dod)

    def _greedy_schedule(self, sat: "Satellite", t: int, global_info: Dict) -> List[int]:
        # 记录本星 DoD 供代价函数使用（能耗发生在本星 n，用 DoD_n(t)），其余逻辑复用 MHSPO
        self._cur_dod = float(sat.dod)
        return super()._greedy_schedule(sat, t, global_info)

    def _local_cost_norm(self, task: "Task", qb_num: float,
                         q_tilde_f: float, d_comp: float, e_comp: float) -> float:
        """与 MHSPO._local_cost_norm 相同，仅把 rho_e·e_comp 换成 rho_batt·(归一化边际 HL)。"""
        cfg = self.cfg
        s_avg = cfg.S_AVG
        s_i_num = task.size / s_avg
        q_tilde_f_num = q_tilde_f / s_avg
        batt = self._lprime(self._cur_dod) * (e_comp / cfg.E_CAP) / (cfg.L_MAX_NEW_RAW + 1e-12)
        raw = (s_i_num * qb_num
               - s_i_num * q_tilde_f_num
               + self.V_lyapunov * (self.rho_d * d_comp + self.rho_e * batt))
        return raw / (self.Q_NORM + 1e-9)

    def _forward_cost_norm(self, task: "Task", b_nm: float, t_nm: float,
                           qf_bits: float, q_tilde_f: float,
                           nb_qb_pred: float) -> float:
        """与 MHSPO._forward_cost_norm 相同，仅把 rho_e·trans_energy 换成 rho_batt·(归一化边际 HL)。"""
        cfg = self.cfg
        s_avg = cfg.S_AVG
        trans_delay = task.size / b_nm + t_nm
        trans_energy = (task.size / b_nm) * cfg.P_T
        batt = self._lprime(self._cur_dod) * (trans_energy / cfg.E_CAP) / (cfg.L_MAX_NEW_RAW + 1e-12)
        s_i_num = task.size / s_avg
        s_max_num = cfg.S_MAX / s_avg
        q_tilde_f_num = q_tilde_f / s_avg
        bracket_num = min(qf_bits - self.theta_n_bits, 0.0) / s_avg
        raw = (self.V_lyapunov * self.rho_d * trans_delay
               + self.V_lyapunov * self.rho_e * batt
               - s_i_num * q_tilde_f_num
               - s_max_num * bracket_num
               + s_i_num * nb_qb_pred)
        return raw / (self.Q_NORM + 1e-9)
