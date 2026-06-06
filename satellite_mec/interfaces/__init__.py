"""
interfaces/__init__.py
======================
公共接口定义（Protocol / ABC）。

使用方式
--------
    from interfaces import PolicyInterface, EnvInterface, MetricsInterface

接口说明
--------
- PolicyInterface  : 所有调度策略必须实现的接口
- EnvInterface     : 仿真环境的对外接口（step / reset / get_*）
- MetricsInterface : 指标记录器的对外接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# PolicyInterface
# ──────────────────────────────────────────────────────────────
class PolicyInterface(ABC):
    """
    所有调度策略的统一接口。

    子类必须实现 get_actions()，可选覆盖其余方法。

    使用示例
    --------
    >>> policy = MyPolicy(config, env)
    >>> obs   = env.get_observations()
    >>> masks = env.get_action_masks()
    >>> actions = policy.get_actions(obs, masks)
    >>> _, _, done, info = env.step(policy=policy)
    """

    needs_training: bool = False  # 是否需要先调用 run_training()
    name: str = "base"

    @abstractmethod
    def get_actions(
        self,
        obs:   Dict[int, List[np.ndarray]],
        masks: Dict[int, List[np.ndarray]],
    ) -> Dict[int, List[int]]:
        """
        根据观测和动作掩码返回每颗卫星的动作序列。

        Parameters
        ----------
        obs   : {sat_id: [state_vec, ...]}  每颗卫星当前 forward_queue 中各任务的状态向量
        masks : {sat_id: [mask_vec, ...]}   对应的合法动作掩码（1=可选，0=非法）

        Returns
        -------
        {sat_id: [action_int, ...]}  每颗卫星、每个任务对应的动作索引
            0          → 本地计算
            1 ~ N_nbr  → 转发给第 k 个邻居

        Raises
        ------
        NotImplementedError : 子类未实现时抛出
        """
        raise NotImplementedError

    def set_eval_mode(self) -> None:
        """切换到评估模式（确定性决策，不记录梯度）。"""

    def set_train_mode(self) -> None:
        """切换到训练模式（随机探索）。"""

    def on_episode_end(self) -> None:
        """每个 rollout episode 结束时的回调（可用于触发更新等）。"""


# ──────────────────────────────────────────────────────────────
# EnvInterface
# ──────────────────────────────────────────────────────────────
class EnvInterface(ABC):
    """
    仿真环境的对外接口。

    使用示例
    --------
    >>> env = SatelliteMECEnv(config)
    >>> env.reset(phase='eval', seeds={'task': 42, 'task_param': 43, 'dod_init': 44})
    >>> for _ in range(5400):
    ...     _, _, done, info = env.step(policy=my_policy)
    """

    @abstractmethod
    def reset(
        self,
        phase: str = "train",
        seeds: Dict[str, int] | None = None,
    ) -> Dict:
        """
        重置环境到初始状态。

        Parameters
        ----------
        phase : 'train' | 'eval' | 'warmup'
        seeds : 随机种子字典，键为 'task' / 'task_param' / 'dod_init'

        Returns
        -------
        空字典（保留接口扩展性）
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        actions: Dict[int, List[int]] | None = None,
        policy: PolicyInterface | None = None,
    ) -> Tuple[Dict, Dict, bool, Dict]:
        """
        执行一个时隙的仿真。

        Parameters
        ----------
        actions : 预先计算好的动作字典（与 policy 二选一）
        policy  : 策略对象，由 env 内部调用 get_actions()（与 actions 二选一）

        Returns
        -------
        next_obs : {sat_id: [state_vec, ...]}
        rewards  : {sat_id: float}
        done     : bool，当前 rollout episode 是否结束
        info     : 包含以下关键字段的字典：
            'slot'               : int    当前时隙编号
            'arrived'            : int    本时隙到达任务数
            'done_tasks'         : int    本时隙完成任务数
            'slot_timeout'       : int    本时隙新增超时任务数
            'episode_timeout'    : int    累计超时任务数
            'completion_rate'    : float  累计完成率
            'avg_dod'            : float  平均 DoD
            'slot_satisfied'     : int    本时隙满意任务数（时延 ≤ deadline）
            'slot_satisfaction_rate' : float  满意度（口径A：分母=完成+超时）
            'slot_e2e_delays'    : list   本时隙各完成任务的端到端时延（秒）
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> Dict[int, List[np.ndarray]]:
        """返回当前时隙各卫星的观测向量列表。"""
        raise NotImplementedError

    @abstractmethod
    def get_action_masks(self) -> Dict[int, List[np.ndarray]]:
        """返回当前时隙各卫星的合法动作掩码列表。"""
        raise NotImplementedError

    @abstractmethod
    def get_critic_obs(self) -> Dict[int, np.ndarray]:
        """返回 MAPPO Critic 所需的全局状态向量（每颗卫星一个）。"""
        raise NotImplementedError

    def get_completion_rate(self) -> float:
        """返回本 episode 的任务完成率（episode_done / episode_arrived）。"""
        raise NotImplementedError

    def get_eval_completion_rate(self) -> float:
        """返回评估阶段的任务完成率（eval_done / eval_arrived）。"""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# MetricsInterface
# ──────────────────────────────────────────────────────────────
class MetricsInterface(ABC):
    """
    指标记录器的对外接口。

    使用示例
    --------
    >>> recorder = MetricsRecorder(config, 'MHSPO')
    >>> recorder.record_slot(info, phase='eval')
    >>> recorder.record_eval_run(run_idx=0)
    >>> summary = recorder.get_summary()
    >>> recorder.save('./results/MHSPO')
    """

    @abstractmethod
    def record_slot(self, info: Dict, phase: str = "train") -> None:
        """记录单个时隙的指标。"""
        raise NotImplementedError

    @abstractmethod
    def record_eval_run(self, run_idx: int) -> None:
        """汇总并归档一次完整评估 run 的指标。"""
        raise NotImplementedError

    @abstractmethod
    def get_summary(self) -> Dict:
        """
        返回所有评估 run 的汇总统计。

        Returns
        -------
        包含以下字段的字典（每个字段为 {'mean', 'std', 'ci95', 'values'}）：
            'completion_rate', 'avg_dod', 'max_dod',
            'avg_health_loss', 'cumulative_health_loss',
            'avg_qf_size', 'avg_z'
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """将所有记录持久化到 output_dir 目录。"""
        raise NotImplementedError
