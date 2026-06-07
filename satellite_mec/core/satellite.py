"""
core/satellite.py
=================
单颗卫星的状态机实现。

使用示例（通常由 Constellation 创建，不直接实例化）
--------
    sat = Satellite(
        sat_id=0,
        neighbors=[1, 5, 20, 24],
        link_rates={1: 200e6, 5: 150e6, 20: 180e6, 24: 220e6},
        prop_delays={1: 0.001, 5: 0.002, 20: 0.003, 24: 0.001},
        solar_seq=np.zeros(T_TOTAL, dtype=np.float32),
        config=cfg,
    )
    sat.reset(rng=np.random.default_rng(42))
    sat.update_solar(t=100)
    sat.admit_task(task)
    sat.init_temp_state()
    reward, forward_info = sat.apply_action(task, action=0, current_slot=100,
                                            neighbor_info={}, lyapunov_calc=calc)
    done_tasks, timeout_tasks = sat.process_tasks(current_slot=100)
    sat.update_dod(nb_start=sat.nb)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from core.dvfs import select_freq as dvfs_select_freq, deadline_floor as dvfs_deadline_floor

if TYPE_CHECKING:
    from core.config import Config
    from core.task import Task
    from core.lyapunov import LyapunovCalculator


class Satellite:
    """
    单颗卫星的完整状态机。

    Attributes（只读，通过方法更新）
    ----------
    sat_id        : 卫星全局编号 [0, N_SATS)
    neighbors     : 4个邻居的 sat_id 列表（顺序：前/后/左/右）
    link_rates    : {neighbor_id: bps}
    prop_delays   : {neighbor_id: s}
    solar_seq     : 预计算太阳能序列 [T_TOTAL]（float32）
    forward_queue : 等待调度的任务队列
    compute_queue : 正在计算的任务队列
    nb            : 当前 compute_queue 中的并发任务数
    dod           : 当前放电深度 [DOD_MIN, DOD_MAX]
    z             : DoD 虚拟队列值
    """

    def __init__(self, sat_id: int, neighbors: List[int],
                 link_rates: Dict[int, float], prop_delays: Dict[int, float],
                 solar_seq: np.ndarray, config: "Config"):
        self.cfg         = config
        self.sat_id      = sat_id
        self.neighbors   = neighbors
        self.link_rates  = link_rates
        self.prop_delays = prop_delays
        self.solar_seq   = solar_seq

        self.forward_queue: List["Task"] = []
        self.compute_queue: List["Task"] = []
        self.qf_size: float = 0.0
        self.qb_size: float = 0.0
        self.nb:      int   = 0
        self.dod:     float = 0.0
        self.z:       float = 0.0
        self.xi:      int   = 1
        self.solar_power: float = 0.0
        self.tau_switch:  int   = 0

        # 预测/暂存状态（每时隙 init_temp_state() 重置）
        self.nb_hat:    int   = 0
        self.z_hat:     float = 0.0
        self.alpha_num: int   = 0
        self.alpha_bar: float = config.ALPHA_BAR_INIT

        self._pending_compute:  List["Task"]              = []
        self._forwarded_tasks:  List[Tuple]               = []
        self.last_done_count:   int                       = 0
        self._comp_energy:      float                     = 0.0
        self._trans_energy:     float                     = 0.0
        self._slot_f_cmp:       float                     = 0.0
        self.last_cpu_freq:     float                     = 0.0   # 上一时隙实际使用的 f_cmp（诊断用）
        self.last_f_floor:      float                     = 0.0   # 上一时隙 deadline floor
        self.q_cycles_hat:      float                     = 0.0   # 本时隙累积 backlog 估计（含已接纳）
        self.f_floor_hat:       float                     = 0.0   # 本时隙累积 floor 估计
        self.slot_health_loss:  float                     = 0.0
        self.slot_delta_l_comp: float                     = 0.0
        self.slot_delta_l_trans: float                    = 0.0
        self.reset()

    # ── 重置 ──────────────────────────────────────────────────
    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        重置卫星至初始状态。

        Parameters
        ----------
        rng : 若提供则用于随机初始化 DoD，否则取配置中间值
        """
        self.forward_queue.clear()
        self.compute_queue.clear()
        self._pending_compute.clear()
        self._forwarded_tasks.clear()
        self.qf_size = 0.0
        self.qb_size = 0.0
        self.nb      = 0
        if rng is not None:
            self.dod = float(rng.uniform(self.cfg.DOD_INIT_LOW, self.cfg.DOD_INIT_HIGH))
        else:
            self.dod = (self.cfg.DOD_INIT_LOW + self.cfg.DOD_INIT_HIGH) / 2
        self.z          = 0.0
        self.alpha_bar  = self.cfg.ALPHA_BAR_INIT
        self.alpha_num  = 0
        self.last_done_count  = 0
        self.nb_hat     = 0
        self.z_hat      = 0.0
        self._comp_energy  = 0.0
        self._trans_energy = 0.0
        self._slot_f_cmp   = 0.0
        self.last_cpu_freq = 0.0
        self.last_f_floor  = 0.0
        self.q_cycles_hat  = 0.0
        self.f_floor_hat   = 0.0
        self.slot_health_loss   = 0.0
        self.slot_delta_l_comp  = 0.0
        self.slot_delta_l_trans = 0.0

    # ── 时隙初始化 ────────────────────────────────────────────
    def update_solar(self, t: int) -> None:
        """更新太阳能状态及日照切换倒计时。"""
        self.solar_power = float(self.solar_seq[t])
        self.xi = 1 if self.solar_power > 0 else 0
        current_xi = self.xi
        self.tau_switch = 0
        max_look = min(t + self.cfg.ORBIT_PERIOD, len(self.solar_seq))
        for future_t in range(t + 1, max_look):
            if (1 if self.solar_seq[future_t] > 0 else 0) != current_xi:
                break
            self.tau_switch += 1

    def init_temp_state(self) -> None:
        """初始化当前时隙的预测状态（每时隙调度前调用一次）。"""
        self.nb_hat = max(self.nb - self.last_done_count, 0)
        self.z_hat  = self.z
        self.alpha_num = 0
        # DVFS 边际能耗估计的累积状态：以当前 compute_queue 为基线
        self.q_cycles_hat = sum(t.get_remaining_size() * t.cpu_cycles
                                for t in self.compute_queue)
        self.f_floor_hat  = self.last_f_floor
        self._comp_energy    = 0.0
        self._trans_energy   = 0.0
        self._slot_f_cmp     = 0.0
        self._forwarded_tasks = []
        self._pending_compute = []

    # ── 任务队列管理 ──────────────────────────────────────────
    def admit_task(self, task: "Task") -> bool:
        """
        尝试将新到达任务加入 forward_queue。

        Returns
        -------
        bool : True=接受，False=队列已满（丢弃）
        """
        if len(self.forward_queue) <= self.cfg.THETA_NUM:
            self.forward_queue.append(task)
            self.qf_size += task.size
            return True
        return False

    def sort_forward_queue(self, current_slot: int) -> None:
        """按剩余时间升序（最紧急优先），相同则按 size 升序排列。"""
        self.forward_queue.sort(key=lambda t: (t.remain_time(current_slot), t.size))

    def remove_timeout_tasks(self, current_slot: int) -> List["Task"]:
        """
        从 forward_queue 移除并标记所有已超时任务。

        Returns
        -------
        list : 被移除的超时任务列表
        """
        timeout, valid = [], []
        for task in self.forward_queue:
            if task.is_timeout(current_slot):
                task.set_timeout()
                timeout.append(task)
            else:
                valid.append(task)
        self.forward_queue = valid
        self.qf_size = sum(t.size for t in self.forward_queue)
        return timeout

    def update_queues(self, new_arrivals: List["Task"],
                      received_tasks: List["Task"],
                      current_slot: int = -1) -> List["Task"]:
        """
        处理新到达和转发接收的任务，提交 _pending_compute 到 compute_queue。

        Parameters
        ----------
        new_arrivals   : 本时隙新到达任务（当前实现不使用，由 admit_task 直接处理）
        received_tasks : 从邻居转发过来的任务
        current_slot   : 若 ≥ 0 则过滤在传输途中已超时的任务

        Returns
        -------
        list : 在传输途中或入队时超时被丢弃的任务列表
        """
        transit_timeout: List["Task"] = []
        for task in received_tasks:
            if current_slot >= 0 and task.is_timeout(current_slot):
                task.set_timeout()
                transit_timeout.append(task)
            else:
                task.status = 'queuing'
                self.forward_queue.append(task)

        cap = self.cfg.MAX_DISPATCH
        for task in self._pending_compute:
            if self.nb < cap:
                self.compute_queue.append(task)
                self.nb += 1
            else:
                task.set_timeout()
                transit_timeout.append(task)

        self.qf_size = sum(t.size for t in self.forward_queue)
        self.qb_size = sum(t.size for t in self.compute_queue)
        self._pending_compute.clear()
        return transit_timeout

    # ── 动作决策 ──────────────────────────────────────────────
    def get_action_mask(self, task: "Task", current_slot: int,
                        neighbor_nb: Dict[int, int]) -> np.ndarray:
        """
        计算任务的合法动作掩码。

        Parameters
        ----------
        task         : 待调度任务
        current_slot : 当前时隙
        neighbor_nb  : {neighbor_id: 当前并发任务数}

        Returns
        -------
        np.ndarray : shape=(action_dim,)，1=合法，0=非法
        """
        cfg  = self.cfg
        mask = np.zeros(cfg.get_action_dim(), dtype=np.float32)
        if self.nb_hat < cfg.MAX_DISPATCH:
            mask[0] = 1.0
        for idx, neighbor_id in enumerate(self.neighbors):
            if task.feasible_forward(
                self.link_rates[neighbor_id], self.prop_delays[neighbor_id],
                neighbor_nb.get(neighbor_id, 0), cfg.CPU_FREQ, cfg.TAU,
                current_slot, cfg.K_MAX
            ):
                mask[idx + 1] = 1.0
        return mask

    def get_state(self, task: "Task", current_slot: int,
                  neighbor_info: Dict) -> np.ndarray:
        """
        构建 Actor 输入的观测向量（54 维）。

        Layout
        ------
            id(1)
          + local(10): qf, qb, nb_hat, dod, z_hat, xi, tau_switch,
                       last_cpu_freq, solar_norm, dod_headroom
          + neighbor(4×9): link_rate, prop_delay, qf, qb, nb, dod, xi,
                           tau_switch, last_cpu_freq
          + task(7): size, cycles, hops, trans_delay, remain,
                     slack_ratio, cycle_rate_need
        """
        cfg       = self.cfg
        id_feat   = np.array([self.sat_id / max(cfg.N_SATS - 1, 1)], dtype=np.float32)
        local_state = np.array([
            self.qf_size / (cfg.Q_F_MAX + 1e-9),
            self.qb_size / (cfg.Q_F_MAX + 1e-9),
            self.nb_hat  / max(cfg.MAX_DISPATCH, 1),
            self.dod     / cfg.DOD_MAX,
            self.z_hat   / max(cfg.Z_MAX, 1e-9),
            float(self.xi),
            self.tau_switch / cfg.ORBIT_PERIOD,
            self.last_cpu_freq / max(cfg.CPU_FREQ, 1.0),
            self.solar_power   / max(cfg.P_SOLAR_MAX, 1e-6),
            max(cfg.DOD_MAX - self.dod, 0.0) / cfg.DOD_MAX,
        ], dtype=np.float32)
        neighbor_state = []
        max_prop = cfg.ORBIT_RADIUS / cfg.SPEED_OF_LIGHT
        for neighbor_id in self.neighbors:
            info = neighbor_info.get(neighbor_id, {})
            neighbor_state.extend([
                self.link_rates.get(neighbor_id, cfg.B_AVG) / cfg.B_MAX,
                self.prop_delays.get(neighbor_id, 0.0) / max(max_prop, 1e-9),
                (info.get('qf_size', 0.0) - cfg.THETA) / (cfg.Q_F_MAX + 1e-9),
                info.get('qb_size', 0.0) / (cfg.Q_F_MAX + 1e-9),
                info.get('nb', 0) / max(cfg.MAX_DISPATCH, 1),
                info.get('dod', 0.0) / cfg.DOD_MAX,
                float(info.get('xi', 1)),
                info.get('tau_switch', 0) / cfg.ORBIT_PERIOD,
                info.get('last_cpu_freq', 0.0) / max(cfg.CPU_FREQ, 1.0),
            ])
        remain = max(task.remain_time(current_slot), 0.0)
        # slack_ratio: 剩余时间 / 估算计算耗时 (>1 表示有余裕，<1 表示赶不上)
        est_comp_time = task.size * task.cpu_cycles / max(cfg.CPU_FREQ, 1.0)
        slack_ratio = remain / max(est_comp_time, 1e-3)
        # cycle_rate_need: 完成所需的最低频率 / f_max
        cycle_rate_need = (task.size * task.cpu_cycles / max(remain, 1e-3)) / max(cfg.CPU_FREQ, 1.0)
        task_state = np.array([
            task.size / cfg.S_MAX, task.cpu_cycles / cfg.H_MAX,
            task.hops / max(cfg.K_MAX, 1), task.trans_delay_acc / cfg.D_MAX_MAX,
            remain / cfg.D_MAX_MAX,
            min(slack_ratio, 10.0) / 10.0,
            min(cycle_rate_need, 1.0),
        ], dtype=np.float32)
        return np.concatenate([id_feat, local_state,
                               np.array(neighbor_state, dtype=np.float32), task_state])

    def get_critic_state(self, neighbor_info: Dict, current_slot: int) -> np.ndarray:
        """构建 Critic 输入的拼接观测向量（每节点 47 维 × 5 节点 = 235 维）。"""
        own_state = self._get_node_state_47()
        neighbor_states = [
            self._get_neighbor_node_state_47(nid, neighbor_info.get(nid, {}))
            for nid in self.neighbors
        ]
        return np.concatenate([own_state] + neighbor_states)

    def _get_node_state_47(self) -> np.ndarray:
        """每节点 critic 子向量（与 Actor 的 (id + own + neighbor) 子集对齐，去掉 task）。"""
        cfg = self.cfg
        id_feat = np.array([self.sat_id / max(cfg.N_SATS - 1, 1)], dtype=np.float32)
        local_state = np.array([
            self.qf_size / (cfg.Q_F_MAX + 1e-9),
            self.qb_size / (cfg.Q_F_MAX + 1e-9),
            self.nb_hat  / max(cfg.MAX_DISPATCH, 1),
            self.dod     / cfg.DOD_MAX,
            self.z_hat   / max(cfg.Z_MAX, 1e-9),
            float(self.xi),
            self.tau_switch / cfg.ORBIT_PERIOD,
            self.last_cpu_freq / max(cfg.CPU_FREQ, 1.0),
            self.solar_power   / max(cfg.P_SOLAR_MAX, 1e-6),
            max(cfg.DOD_MAX - self.dod, 0.0) / cfg.DOD_MAX,
        ], dtype=np.float32)
        return np.concatenate([id_feat, local_state,
                               np.zeros(cfg.N_NEIGHBORS * 9, dtype=np.float32)])

    def _get_neighbor_node_state_47(self, neighbor_id: int, info: Dict) -> np.ndarray:
        cfg = self.cfg
        id_feat = np.array([neighbor_id / max(cfg.N_SATS - 1, 1)], dtype=np.float32)
        local_state = np.array([
            info.get('qf_size', 0.0) / (cfg.Q_F_MAX + 1e-9),
            info.get('qb_size', 0.0) / (cfg.Q_F_MAX + 1e-9),
            info.get('nb', 0) / max(cfg.MAX_DISPATCH, 1),
            info.get('dod', 0.0) / cfg.DOD_MAX,
            0.0,
            float(info.get('xi', 1)),
            info.get('tau_switch', 0) / cfg.ORBIT_PERIOD,
            info.get('last_cpu_freq', 0.0) / max(cfg.CPU_FREQ, 1.0),
            info.get('solar_power', 0.0) / max(cfg.P_SOLAR_MAX, 1e-6),
            max(cfg.DOD_MAX - info.get('dod', 0.0), 0.0) / cfg.DOD_MAX,
        ], dtype=np.float32)
        return np.concatenate([id_feat, local_state,
                               np.zeros(cfg.N_NEIGHBORS * 9, dtype=np.float32)])

    def apply_action(self, task: "Task", action: int, current_slot: int,
                     neighbor_info: Dict,
                     lyapunov_calc: "LyapunovCalculator") -> Tuple[float, Optional[Tuple]]:
        """
        执行调度动作。

        Parameters
        ----------
        task           : 待调度任务
        action         : 0=本地计算，1~N=转发给第(action-1)个邻居
        current_slot   : 当前时隙
        neighbor_info  : {neighbor_id: info_dict}
        lyapunov_calc  : 代价计算器实例

        Returns
        -------
        reward       : float，= -cost
        forward_info : (target_sat_id, task) 若转发，否则 None
        """
        forward_info = None
        sat_state    = lyapunov_calc.get_sat_state(self)

        if action == 0:  # 本地计算
            nb_next = self.nb_hat + 1
            cost = lyapunov_calc.normalized_local_cost(task, sat_state, nb_next, current_slot)
            # 用更新前的 sat_state 计算 delta_dod_comp，再推进 hat
            delta_dod_local = lyapunov_calc.delta_dod_comp(task, sat_state)
            self.nb_hat    += 1
            self.z_hat     += delta_dod_local
            self.alpha_num += 1
            # 推进 DVFS 累积估计（供同时隙后续任务的差分使用）
            task_cycles = task.size * task.cpu_cycles
            self.q_cycles_hat += task_cycles
            new_floor_task = nb_next * task_cycles / max(task.deadline, self.cfg.TAU)
            self.f_floor_hat = max(self.f_floor_hat, new_floor_task)
            if nb_next > 0:
                # Li-style DVFS：单任务能耗无闭式，记录上界估计供监控用
                self._comp_energy += (self.cfg.KAPPA * task.size * task.cpu_cycles
                                      * (self.cfg.CPU_FREQ ** 2))
            self._pending_compute.append(task)
            task.set_computing()
            task.current_sat = self.sat_id
            self._remove_from_forward_queue(task)

        else:  # 转发
            neighbor_idx = action - 1
            neighbor_id  = self.neighbors[neighbor_idx]
            b_nm = self.link_rates[neighbor_id]
            t_nm = self.prop_delays[neighbor_id]
            neighbor_state = lyapunov_calc.get_neighbor_state(neighbor_info.get(neighbor_id, {}))
            cost = lyapunov_calc.normalized_forward_cost(task, sat_state, neighbor_state,
                                                          b_nm, current_slot)
            self.z_hat       += lyapunov_calc.delta_dod_trans(task, b_nm)
            self._trans_energy += self.cfg.P_T * task.size / b_nm
            task.forward(b_nm=b_nm, t_nm=t_nm)
            task.current_sat = neighbor_id
            self._forwarded_tasks.append((neighbor_id, task, b_nm))
            forward_info = (neighbor_id, task)
            self._remove_from_forward_queue(task)

        return -cost, forward_info

    # ── 计算推进 ──────────────────────────────────────────────
    def process_tasks(self, current_slot: int = -1) -> Tuple[List["Task"], List["Task"]]:
        """
        推进 compute_queue 中所有任务的计算进度一个时隙。

        步骤：(1) 驱逐超时任务；(2) 按当前 backlog 通过 Li-style DVFS 求解
        f_cmp(t) ∈ [0, CPU_FREQ]；(3) 把 f_cmp·τ 的 cycle 预算平分给剩余任务。

        f_cmp 由 core.dvfs.select_freq 返回，记入 self._slot_f_cmp 供 update_dod
        计算真实能耗使用。

        Parameters
        ----------
        current_slot : 若 ≥ 0 则启用超时驱逐

        Returns
        -------
        done_tasks    : 本时隙完成的任务列表
        timeout_tasks : 本时隙在 compute_queue 中超时被驱逐的任务列表
        """
        cfg = self.cfg
        done_tasks:    List["Task"] = []
        timeout_tasks: List["Task"] = []

        # 第一步：驱逐超时任务
        if current_slot >= 0:
            alive, expired = [], []
            for task in self.compute_queue:
                if task.is_timeout(current_slot):
                    task.set_timeout(); expired.append(task)
                else:
                    alive.append(task)
            if expired:
                self.compute_queue = alive
                self.nb -= len(expired)
                timeout_tasks.extend(expired)

        if self.nb == 0:
            self._slot_f_cmp     = 0.0
            self.last_cpu_freq   = 0.0
            self.last_f_floor    = 0.0
            self.last_done_count = 0
            return done_tasks, timeout_tasks

        # 第二步：DVFS 选频（Lyapunov 闭式 + 截止时间下限）
        q_cycles = sum(t.get_remaining_size() * t.cpu_cycles for t in self.compute_queue)
        f_floor  = dvfs_deadline_floor(cfg, self.compute_queue, current_slot)
        f_cmp    = dvfs_select_freq(cfg, q_cycles, f_floor=f_floor)
        self._slot_f_cmp   = f_cmp
        self.last_cpu_freq = f_cmp
        self.last_f_floor  = f_floor

        # 第三步：平分 cycle 预算推进任务
        cycles_per_task = f_cmp * cfg.TAU / self.nb if self.nb > 0 else 0.0
        tasks_to_remove = []
        for task in self.compute_queue:
            processed = min(cycles_per_task / task.cpu_cycles,
                            task.get_remaining_size())
            task.update_processed(processed)
            if task.is_done():
                task.finish_slot = current_slot
                tasks_to_remove.append(task)
                done_tasks.append(task)
        for task in tasks_to_remove:
            self.compute_queue.remove(task)
            self.nb -= 1
        self.qb_size         = sum(t.size for t in self.compute_queue)
        self.last_done_count = len(done_tasks)
        return done_tasks, timeout_tasks

    # ── DoD 更新 ──────────────────────────────────────────────
    def update_dod(self, nb_start: int = -1) -> None:
        """
        更新 DoD 和虚拟队列 z（每时隙末尾调用一次）。

        计算能耗采用 Li-style 整星 DVFS：使用 process_tasks 求得的 f_cmp(t)
        计算 P_comp = κ·f_cmp³；nb_start 不再参与能耗（保留参数为兼容）。

        Parameters
        ----------
        nb_start : 保留参数（用于其它统计），不再参与能耗计算
        """
        cfg       = self.cfg
        dod_before = self.dod

        f_cmp = self._slot_f_cmp
        delta_comp = (cfg.TAU * cfg.KAPPA * (f_cmp ** 3) / cfg.E_CAP
                      if f_cmp > 0.0 else 0.0)
        delta_trans = (cfg.P_T * sum(
            task.size / b_nm for _, task, b_nm in self._forwarded_tasks if b_nm > 0
        ) / cfg.E_CAP)
        delta_house = cfg.P_HOUSEKEEPING * cfg.TAU / cfg.E_CAP
        delta_solar_raw = self.solar_power * cfg.TAU / cfg.E_CAP
        delta_solar = min(delta_solar_raw, max(dod_before - cfg.DOD_MIN, 0.0))
        delta = delta_comp + delta_trans + delta_house - delta_solar

        a = cfg.A_COEF
        l_prime = ((10 ** (a * (dod_before - 1)))
                   * (1.0 + a * math.log(10) * dod_before))
        self.slot_delta_l_comp  = l_prime * delta_comp
        self.slot_delta_l_trans = l_prime * delta_trans
        self.slot_health_loss   = self.slot_delta_l_comp + self.slot_delta_l_trans

        self.dod = float(np.clip(dod_before + delta, cfg.DOD_MIN, cfg.DOD_MAX))
        self.z   = max(self.z + delta, 0.0)

    def update_alpha_avg(self) -> None:
        """更新 alpha_bar（滑动平均并发数估计）。"""
        self.alpha_bar = (1 - self.cfg.MU) * self.alpha_bar + self.cfg.MU * self.alpha_num

    # ── 查询接口 ──────────────────────────────────────────────
    def get_forwarded_tasks(self) -> List[Tuple]:
        """返回本时隙已转发任务的记录列表 [(neighbor_id, task, b_nm), ...]。"""
        return self._forwarded_tasks

    def get_info(self) -> Dict:
        """返回供邻居卫星参考的精简状态字典。"""
        return {
            'qf_size':       self.qf_size,
            'qb_size':       self.qb_size,
            'n_f':           len(self.forward_queue),
            'nb':            self.nb,
            'dod':           self.dod,
            'xi':            self.xi,
            'tau_switch':    self.tau_switch,
            'last_cpu_freq': self.last_cpu_freq,
            'solar_power':   self.solar_power,
        }

    def _remove_from_forward_queue(self, task: "Task") -> None:
        try:
            self.forward_queue.remove(task)
            self.qf_size = max(self.qf_size - task.size, 0.0)
        except ValueError:
            pass

    def __repr__(self) -> str:
        return (f"Satellite(id={self.sat_id}, "
                f"QF={self.qf_size/1e6:.1f}MB, nb={self.nb}, DoD={self.dod:.3f})")
