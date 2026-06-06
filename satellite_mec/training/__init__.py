"""
training/__init__.py
====================
训练模块公共 API。

对外暴露
--------
    MAPPOPolicy        : MAPPO 在线策略（实现 PolicyInterface）
    create_mappo_no_dod: 创建无 DoD 惩罚的 MAPPO 变体
    MAPPOTrainer       : PPO 更新 + 模型保存/加载
    Actor              : 策略网络
    Critic             : 价值网络
    RolloutBuffer      : 轨迹经验缓冲区

快速上手
--------
    from training import MAPPOPolicy, create_mappo_no_dod

    # 标准 MAPPO
    mappo = MAPPOPolicy(cfg, name='MAPPO')

    # 禁用 DoD 惩罚的消融版本
    mappo_no_dod = create_mappo_no_dod(cfg, env)

    # 训练（由 ExperimentRunner 编排）
    for t in range(cfg.T_TRAIN):
        rewards, done, info = mappo.run_step(env)

    # 保存 / 加载
    mappo.save('./checkpoints/MAPPO')
    mappo.load('./checkpoints/MAPPO')
"""

from .policy  import MAPPOPolicy, create_mappo_no_dod
from .trainer import MAPPOTrainer
from .networks import Actor, Critic, RolloutBuffer

__all__ = [
    'MAPPOPolicy',
    'create_mappo_no_dod',
    'MAPPOTrainer',
    'Actor',
    'Critic',
    'RolloutBuffer',
]
