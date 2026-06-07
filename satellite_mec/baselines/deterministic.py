"""
baselines/deterministic.py
===========================
三种确定性基线调度策略。

所有策略均实现 PolicyInterface，可直接传入 env.step(policy=...) 使用。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from core.lyapunov import LyapunovCalculator
from interfaces import PolicyInterface

if TYPE_CHECKING:
    from core.config import Config
    from core.env import SatelliteMECEnv


class LocalOnlyPolicy(PolicyInterface):
    """
    本地优先策略（Local Only Baseline）。

    逻辑
    ----
    严格本地处理：所有任务一律选动作0（本地计算），即使本地不可行。
    对应参考论文 (Zhang et al., TMC 2024) 中的 LSO：
        "For each task, the access satellite will process it locally."

    本地不可行时任务排队等待，超过 deadline 由环境标记为 timeout。
    """

    needs_training: bool = False
    name: str = 'LocalOnly'

    def __init__(self, config: "Config", env: "SatelliteMECEnv"):
        self.cfg = config
        self.env = env

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        actions = {}
        sats    = self.env.constellation.satellites
        for sat in sats:
            actions[sat.sat_id] = [0] * len(sat.forward_queue)
        return actions


class GreedyDelayPolicy(PolicyInterface):
    """
    贪心最小时延策略（Greedy Delay Baseline）。

    逻辑
    ----
    对每个任务，枚举所有合法动作，选择估算完成时延最小的一个：
    - 动作0：est_comp_time(task, nb_hat_sim)
    - 动作k：size/b_nm + t_nm + est_comp_time(task, nb_m)
    """

    needs_training: bool = False
    name: str = 'GreedyDelay'

    def __init__(self, config: "Config", env: "SatelliteMECEnv"):
        self.cfg = config
        self.env = env

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        cfg         = self.cfg
        actions     = {}
        sats        = self.env.constellation.satellites
        global_info = self.env._global_info

        for sat in sats:
            n         = sat.sat_id
            sat_masks = masks.get(n, [])
            sat_actions: List[int] = []
            nb_hat_sim = sat.nb_hat

            for task, mask in zip(sat.forward_queue, sat_masks):
                best_action, best_time = -1, float('inf')
                if mask[0] > 0:
                    ct = self._est_comp_time(task, nb_hat_sim, cfg.CPU_FREQ, cfg.TAU)
                    if ct < best_time:
                        best_time, best_action = ct, 0
                for idx, neighbor_id in enumerate(sat.neighbors):
                    if mask[idx + 1] > 0:
                        b_nm = sat.link_rates[neighbor_id]
                        t_nm = sat.prop_delays[neighbor_id]
                        nb_m = global_info.get(neighbor_id, {}).get('nb', 0)
                        total = (task.size / b_nm + t_nm
                                 + self._est_comp_time(task, nb_m, cfg.CPU_FREQ, cfg.TAU))
                        if total < best_time:
                            best_time, best_action = total, idx + 1
                if best_action == -1:
                    best_action = 0
                if best_action == 0:
                    nb_hat_sim += 1
                sat_actions.append(best_action)

            actions[n] = sat_actions
        return actions

    @staticmethod
    def _est_comp_time(task, nb: int, cpu_freq: float, tau: float) -> float:
        nb = max(nb, 0)
        slots = math.ceil(task.size * task.cpu_cycles / (cpu_freq * tau))
        return slots * tau


class LyapunovGreedyPolicy(PolicyInterface):
    """
    Lyapunov 代价贪心策略（LyapunovGreedy Baseline）。

    逻辑
    ----
    对每个任务，枚举所有合法动作，选择 Lyapunov 归一化代价最小的动作。
    同时维护 nb_hat / z_hat 的滚动更新，与环境保持一致。

    说明
    ----
    该策略与 MAPPO 使用相同的代价函数，但采用贪心确定性决策
    而不是神经网络策略，可作为 MAPPO 上限的参考基准。
    """

    needs_training: bool = False
    name: str = 'LyapunovGreedy'

    def __init__(self, config: "Config", env: "SatelliteMECEnv"):
        self.cfg          = config
        self.env          = env
        self.lyapunov_calc = LyapunovCalculator(config)

    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        actions     = {}
        sats        = self.env.constellation.satellites
        global_info = self.env._global_info

        for sat in sats:
            n         = sat.sat_id
            sat_masks = masks.get(n, [])
            sat_actions: List[int] = []
            sat_state  = self.lyapunov_calc.get_sat_state(sat)

            for task, mask in zip(sat.forward_queue, sat_masks):
                if mask.sum() == 0:
                    sat_actions.append(0)
                    continue
                best_action, best_cost = -1, float('inf')
                if mask[0] > 0:
                    cost = self.lyapunov_calc.normalized_local_cost(
                        task, sat_state, sat.nb_hat + 1, self.env.current_slot)
                    if cost < best_cost:
                        best_cost, best_action = cost, 0
                for idx, neighbor_id in enumerate(sat.neighbors):
                    if mask[idx + 1] > 0:
                        b_nm = sat.link_rates[neighbor_id]
                        nb_state = self.lyapunov_calc.get_neighbor_state(
                            global_info.get(neighbor_id, {}))
                        cost = self.lyapunov_calc.normalized_forward_cost(
                            task, sat_state, nb_state, b_nm, self.env.current_slot)
                        if cost < best_cost:
                            best_cost, best_action = cost, idx + 1
                if best_action == -1:
                    best_action = int(np.argmax(mask))

                # 同步更新 sat 的临时状态（与 apply_action 保持一致）
                if best_action == 0:
                    pre_state = self.lyapunov_calc.get_sat_state(sat)
                    delta_dod_local = self.lyapunov_calc.delta_dod_comp(task, pre_state)
                    sat.nb_hat    += 1
                    sat.z_hat     += delta_dod_local
                    sat.alpha_num += 1
                    task_cycles = task.size * task.cpu_cycles
                    sat.q_cycles_hat += task_cycles
                    new_floor_task = sat.nb_hat * task_cycles / max(task.deadline, self.cfg.TAU)
                    sat.f_floor_hat  = max(sat.f_floor_hat, new_floor_task)
                else:
                    neighbor_id = sat.neighbors[best_action - 1]
                    b_nm = sat.link_rates[neighbor_id]
                    sat.z_hat += self.lyapunov_calc.delta_dod_trans(task, b_nm)

                sat_actions.append(best_action)

            actions[n] = sat_actions
        return actions
