"""
core/task.py
============
卫星MEC任务数据类。

使用示例
--------
    task = Task(
        task_id=(0, 3, 1),
        size=20e6,          # bits
        cpu_cycles=200.0,   # cycles/bit
        deadline=8.0,       # s
        arrive_slot=10,
        access_sat=3,
    )
    remaining = task.remain_time(current_slot=15)
    is_late   = task.is_timeout(current_slot=15)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Task:
    """
    表示一个计算卸载任务。

    Attributes
    ----------
    task_id       : (arrive_slot, sat_id, seq_idx)，全局唯一
    size          : bits，任务数据量（同时作为计算量单位的分子）
    cpu_cycles    : cycles/bit，每bit所需CPU周期
    deadline      : s，相对截止时间（从到达时刻起算）
    arrive_slot   : 任务到达时隙编号
    access_sat    : 初始接入卫星ID
    tau           : s，时隙长度（默认1.0）
    hops          : 已转发跳数
    trans_delay_acc : s，累积传输时延（含传播时延）
    current_sat   : 当前所在卫星ID
    processed     : bits，已完成计算量
    status        : 'queuing' | 'computing' | 'done' | 'timeout'
    finish_slot   : 完成计算的时隙编号（-1表示未完成）
    """

    task_id:     Tuple[int, int, int]
    size:        float
    cpu_cycles:  float
    deadline:    float
    arrive_slot: int
    access_sat:  int
    tau:         float = 1.0

    hops:             int   = field(default=0)
    trans_delay_acc:  float = field(default=0.0)
    current_sat:      int   = field(default=-1)
    processed:        float = field(default=0.0)
    status:           str   = field(default='queuing')
    finish_slot:      int   = field(default=-1)

    def __post_init__(self):
        if self.current_sat == -1:
            self.current_sat = self.access_sat

    # ── 时间相关 ──────────────────────────────────────────────
    def remain_time(self, current_slot: int) -> float:
        """
        计算当前时隙的剩余可用时间（秒）。

        Parameters
        ----------
        current_slot : 当前时隙编号

        Returns
        -------
        float : deadline - elapsed_time - trans_delay_acc
            可能为负值（表示已超时）
        """
        elapsed = (current_slot - self.arrive_slot) * self.tau
        return self.deadline - elapsed - self.trans_delay_acc

    def is_timeout(self, current_slot: int) -> bool:
        """当 remain_time ≤ 0 时返回 True。"""
        return self.remain_time(current_slot) <= 0.0

    def is_done(self) -> bool:
        """已完成计算（processed ≥ size）。"""
        return self.processed >= self.size - 1e-9

    # ── 状态更新 ──────────────────────────────────────────────
    def forward(self, b_nm: float, t_nm: float) -> None:
        """
        记录一次转发操作，累积传输时延。

        Parameters
        ----------
        b_nm : bits/s，当前链路速率
        t_nm : s，传播时延
        """
        self.hops += 1
        self.trans_delay_acc += self.size / b_nm + t_nm

    def update_processed(self, amount: float) -> None:
        """
        增加已完成计算量，达到 size 时自动标记为 done。

        Parameters
        ----------
        amount : bits，本时隙完成的计算量
        """
        self.processed = min(self.processed + amount, self.size)
        if self.is_done():
            self.status = 'done'

    def set_timeout(self)    -> None: self.status = 'timeout'
    def set_computing(self)  -> None: self.status = 'computing'

    # ── 可行性判断 ────────────────────────────────────────────
    def est_comp_delay(self, nb_m: int, cpu_freq_m: float, tau: float) -> float:
        """
        估算在目标卫星（nb_m+1个并发任务）上的计算时延。

        Parameters
        ----------
        nb_m       : 目标卫星当前并发任务数（本任务到达前）
        cpu_freq_m : cycles/s，目标卫星CPU频率
        tau        : s，时隙长度

        Returns
        -------
        float : 估算完成时延（秒），无法完成时返回 inf
        """
        if cpu_freq_m <= 0 or nb_m < 0:
            return float('inf')
        slots = math.ceil(self.size * self.cpu_cycles / (cpu_freq_m * tau))
        return slots * tau

    def feasible_forward(self, b_nm: float, t_nm: float, nb_m: int,
                         cpu_freq_m: float, tau: float,
                         current_slot: int, k_max: int) -> bool:
        """
        判断转发到邻居是否在截止时间内可完成（含传输+计算）。

        Returns
        -------
        bool : True 表示可行
        """
        if self.hops >= k_max:
            return False
        trans_time = self.size / b_nm + t_nm
        comp_delay = self.est_comp_delay(nb_m, cpu_freq_m, tau)
        return trans_time + comp_delay <= self.remain_time(current_slot)

    def feasible_local(self, nb_hat: int, cpu_freq_n: float,
                       tau: float, current_slot: int) -> bool:
        """
        判断本地计算是否在截止时间内可完成。

        Parameters
        ----------
        nb_hat : 当前预测并发数（已含本任务之前已调度的任务数）
        """
        return self.est_comp_delay(nb_hat, cpu_freq_n, tau) <= self.remain_time(current_slot)

    def get_remaining_size(self) -> float:
        """返回剩余未完成的计算量（bits）。"""
        return max(self.size - self.processed, 0.0)

    def __repr__(self) -> str:
        return (f"Task(id={self.task_id}, size={self.size/1e6:.1f}MB, "
                f"hops={self.hops}, status={self.status})")

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        return isinstance(other, Task) and self.task_id == other.task_id
