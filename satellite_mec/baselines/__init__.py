"""
baselines/__init__.py
=====================
所有基线（非学习型）调度策略。

对外暴露
--------
    LocalOnlyPolicy     — 本地优先，不可行时转发
    GreedyDelayPolicy   — 贪心最小时延（本地 vs 转发逐一比较）
    LyapunovGreedyPolicy — Lyapunov 代价贪心（等同于 MAPPO 的确定性基准）
    MHSPOPolicy         — Zhang et al. TMC 2024 论文复现
    DOGDPredictor       — MHSPO 所用的工作负载预测器
    GDCOPolicy          — Chen et al. TMC 2025 博弈论分布式卸载复现（势博弈 + NE）

使用示例
--------
    from core import Config, SatelliteMECEnv
    from baselines import GreedyDelayPolicy, MHSPOPolicy

    cfg    = Config()
    env    = SatelliteMECEnv(cfg)
    policy = GreedyDelayPolicy(cfg, env)

    env.reset(phase='eval')
    for _ in range(5400):
        _, _, _, info = env.step(policy=policy)
"""

from baselines.deterministic import (
    LocalOnlyPolicy,
    GreedyDelayPolicy,
    LyapunovGreedyPolicy,
)
from baselines.mhspo import MHSPOPolicy, DOGDPredictor
from baselines.gdco import GDCOPolicy

__all__ = [
    "LocalOnlyPolicy",
    "GreedyDelayPolicy",
    "LyapunovGreedyPolicy",
    "MHSPOPolicy",
    "DOGDPredictor",
    "GDCOPolicy",
]
