"""
evaluation/__init__.py
======================
指标记录、实验编排工具。

对外暴露
--------
    MetricsRecorder  — 逐时隙/逐run/汇总统计记录器
    ExperimentRunner — 实验流程编排（预热/训练/评估/保存）
"""

from evaluation.recorder import MetricsRecorder
from evaluation.runner import ExperimentRunner

__all__ = ["MetricsRecorder", "ExperimentRunner"]
