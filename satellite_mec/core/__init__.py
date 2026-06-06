"""
core/__init__.py
================
核心仿真组件。

对外暴露
--------
    Config, Task, LyapunovCalculator, Satellite, Constellation, SatelliteMECEnv

使用示例
--------
    from core import Config, SatelliteMECEnv, LyapunovCalculator

    cfg = Config()
    env = SatelliteMECEnv(cfg)
    env.reset(phase='eval', seeds={'task': 42, 'task_param': 43, 'dod_init': 44})
    for _ in range(5400):
        _, _, done, info = env.step(policy=my_policy)
"""

# 按依赖顺序导入，避免循环引用
from core.config import Config, TrainConfig, AblationConfig, SensitivityVConfig, LoadTestConfig
from core.task import Task
from core.lyapunov import LyapunovCalculator
from core.satellite import Satellite
from core.constellation import Constellation
from core.env import SatelliteMECEnv

__all__ = [
    "Config",
    "TrainConfig",
    "AblationConfig",
    "SensitivityVConfig",
    "LoadTestConfig",
    "Task",
    "LyapunovCalculator",
    "Satellite",
    "Constellation",
    "SatelliteMECEnv",
]
