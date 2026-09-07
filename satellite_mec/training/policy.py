"""
training/policy.py
==================
MAPPOPolicy：实现 PolicyInterface 的 MAPPO 在线策略。

职责
----
    - 封装 Actor/Critic 的前向推断（评估/训练两种模式）
    - 在训练模式下收集 Critic 价值和 Actor 经验
    - 在 rollout episode 结束时自动触发 PPO 更新
    - 提供 quick_eval()、save()、load() 等工具方法

依赖
----
    interfaces          → PolicyInterface
    core.config         → Config
    core.lyapunov       → LyapunovCalculator
    training.networks   → RolloutBuffer
    training.trainer    → MAPPOTrainer

使用示例
--------
    from training import MAPPOPolicy, create_mappo_no_dod

    mappo = MAPPOPolicy(cfg, name='MAPPO')
    # 训练阶段：runner 会调用 run_step(env) 完成 收集+更新
    for t in range(cfg.T_TRAIN):
        rewards, done, info = mappo.run_step(env)

    # 评估阶段
    mappo.set_eval_mode()
    env.reset(phase='eval', seeds=...)
    for _ in range(cfg.T_EVAL):
        _, _, done, info = env.step(policy=mappo)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from interfaces import PolicyInterface
from core.lyapunov import LyapunovCalculator
from .networks import RolloutBuffer
from .trainer import MAPPOTrainer


class MAPPOPolicy(PolicyInterface):
    """
    MAPPO 策略（实现 PolicyInterface）。

    参数
    ----
    config        : Config
    lyapunov_calc : LyapunovCalculator（可选，默认从 config 构造）
    name          : 策略名称，用于日志和文件命名

    属性
    ----
    needs_training : True（需要先调用 ExperimentRunner.run_training()）
    trainer        : MAPPOTrainer（包含 Actor / Critic / 优化器）
    buffer         : RolloutBuffer（episode 经验缓冲区）
    learning_curve : 快速评估历史记录 [{update, completion_rate, ...}]
    """

    needs_training: bool = True

    def __init__(
        self,
        config,
        lyapunov_calc: Optional[LyapunovCalculator] = None,
        name: str = 'MAPPO',
    ):
        self.cfg           = config
        self.name          = name
        self.lyapunov_calc = lyapunov_calc or LyapunovCalculator(config)
        self.trainer       = MAPPOTrainer(config)
        self.buffer        = RolloutBuffer(config)
        self.actor         = self.trainer.actor
        self.critic        = self.trainer.critic

        self.learning_curve:   List[Dict] = []
        self._slot_inference:  Dict[int, List[Tuple]] = {}
        self._slot_critic:     Dict[int, Tuple] = {}
        self._current_slot_t:  int  = 0
        self._eval_mode:       bool = False

    # ── PolicyInterface 必须实现 ──────────────────────────────────────────────

    def get_actions(
        self,
        obs:   Dict[int, List[np.ndarray]],
        masks: Dict[int, List[np.ndarray]],
    ) -> Dict[int, List[int]]:
        """Batch 接口（保留兼容 baseline 用，但 MAPPO 训练走 act_one）。"""
        actions = {}
        self._slot_inference = {}
        for n in range(self.cfg.N_SATS):
            sat_obs   = obs.get(n, [])
            sat_masks = masks.get(n, [])
            sat_actions, sat_inference = [], []
            for state, mask in zip(sat_obs, sat_masks):
                state_t = torch.FloatTensor(state).to(self.trainer.device)
                mask_t  = torch.FloatTensor(mask).to(self.trainer.device)
                action, log_prob, _ = self.actor.get_action(
                    state_t, mask_t, deterministic=self._eval_mode)
                sat_actions.append(action)
                sat_inference.append((state, action, log_prob, mask))
            actions[n] = sat_actions
            self._slot_inference[n] = sat_inference
        return actions

    def act_one(
        self,
        state: np.ndarray,
        mask:  np.ndarray,
    ) -> Tuple[int, float]:
        """单任务决策（sequential 路径调用）。返回 (action, log_prob)。"""
        state_t = torch.FloatTensor(state).to(self.trainer.device)
        mask_t  = torch.FloatTensor(mask).to(self.trainer.device)
        action, log_prob, _ = self.actor.get_action(
            state_t, mask_t, deterministic=self._eval_mode)
        return action, log_prob

    def record_task_transition(
        self,
        sat_id:      int,
        slot_t:      int,
        state:       np.ndarray,
        action:      int,
        log_prob:    float,
        mask:        np.ndarray,
        task_reward: float = 0.0,
    ) -> None:
        """sequential 路径：env.step 在 apply_action 后调用，写入 buffer。"""
        if self._eval_mode:
            return
        self.buffer.add_task(sat_id, slot_t, state, action, log_prob, mask,
                             task_reward=task_reward)

    def set_eval_mode(self) -> None:
        """切换到评估模式（确定性贪婪，不收集经验）。"""
        self._eval_mode = True
        self.actor.eval()
        self.critic.eval()

    def set_train_mode(self) -> None:
        """切换到训练模式（随机采样，收集经验）。"""
        self._eval_mode = False
        self.actor.train()
        self.critic.train()

    def on_episode_end(self) -> None:
        """rollout episode 结束时的回调（目前由 run_step 内部处理，此处保留扩展）。"""

    # ── 训练辅助方法 ──────────────────────────────────────────────────────────

    def collect_critic_values(self, env) -> None:
        """在当前时隙调用 env 之前，提前记录各卫星的 Critic 价值。"""
        self._slot_critic = {}
        for n, critic_state in env.get_critic_obs().items():
            state_t = torch.FloatTensor(critic_state).to(self.trainer.device)
            with torch.no_grad():
                value = self.critic.get_value(state_t)
            self._slot_critic[n] = (critic_state, float(value.item()))

    def store_slot_data(
        self,
        t:          int,
        rewards:    Dict[int, float],
        n_executed: Dict[int, int],
        done:       bool,
    ) -> None:
        """将当前时隙的经验写入 RolloutBuffer。"""
        for n in range(self.cfg.N_SATS):
            if n in self._slot_critic:
                critic_state, value = self._slot_critic[n]
                self.buffer.add_slot(n, t, critic_state, rewards.get(n, 0.0), value, done)
            n_exec = n_executed.get(n, 0)
            for state, action, log_prob, mask in self._slot_inference.get(n, [])[:n_exec]:
                self.buffer.add_task(n, t, state, action, log_prob, mask)

    def run_step(self, env) -> Tuple[Dict, bool, Dict]:
        """
        执行一个完整的仿真时隙（含经验收集，episode 结束时触发 PPO 更新）。

        Parameters
        ----------
        env : SatelliteMECEnv

        Returns
        -------
        (rewards, done, info)
        """
        t = env.current_slot
        if not self._eval_mode:
            self.collect_critic_values(env)
        _, rewards, done, info = env.step(policy=self)
        if not self._eval_mode:
            self.store_slot_data(t, rewards, info.get('n_executed', {}), done)
            if done:
                self._trigger_update(env)
        return rewards, done, info

    def _trigger_update(self, env) -> Optional[Dict]:
        if self.buffer.size() == 0:
            return None
        bootstrap = self.trainer.compute_bootstrap(env, self.buffer)
        metrics   = self.trainer.update(self.buffer, bootstrap)
        self.buffer.clear()
        if metrics:
            self.trainer.print_metrics(metrics)
        return metrics

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def quick_eval(self, env, n_slots: int = 3600) -> Tuple[float, float, float]:
        """
        在独立的评估种子上快速评估当前策略（不污染训练 RNG）。

        Returns
        -------
        (completion_rate, avg_dod, avg_health_loss)
        """
        cfg = self.cfg
        # 保存训练 RNG 状态
        rng_task  = env.constellation.rng_task.bit_generator.state
        rng_param = env.constellation.rng_task_param.bit_generator.state

        self.set_eval_mode()
        env.reset(phase='eval', seeds=cfg.get_quick_eval_seeds())
        total_dod, total_hl = [], []
        for _ in range(n_slots):
            _, done, info = self.run_step(env)
            total_dod.append(info['avg_dod'])
            total_hl.append(info.get('avg_health_loss', 0.0))

        completion_rate = env.get_eval_completion_rate()
        avg_dod = float(np.mean(total_dod))
        avg_hl  = float(np.mean(total_hl))

        self.set_train_mode()
        # 恢复训练 RNG 状态
        env.constellation.rng_task.bit_generator.state = rng_task
        env.constellation.rng_task_param.bit_generator.state = rng_param
        env.reset(phase='train')

        self.learning_curve.append({
            'update':           self.trainer.update_count,
            'completion_rate':  completion_rate,
            'avg_dod':          avg_dod,
            'avg_health_loss':  avg_hl,
        })
        print(
            f"[QuickEval] update={self.trainer.update_count}, "
            f"CR={completion_rate:.3f}, DoD={avg_dod:.4f}, HL={avg_hl:.4e}"
        )
        return completion_rate, avg_dod, avg_hl

    def save(self, path: str, extra_info: Optional[Dict] = None) -> None:
        """
        持久化模型（委托给 trainer.save，附加 learning_curve）。

        保存内容
        --------
            actor.pth, critic.pth, actor_old.pth,
            actor_optimizer.pth, critic_optimizer.pth,
            progress.pth, learning_curve.json
        """
        os.makedirs(path, exist_ok=True)
        extra = extra_info or {}
        extra['learning_curve'] = self.learning_curve
        self.trainer.save(path, extra)
        with open(os.path.join(path, 'learning_curve.json'), 'w', encoding='utf-8') as f:
            json.dump(self.learning_curve, f, indent=2, ensure_ascii=False)
        print(f"[{self.name}] 模型已保存到 {path}")

    def load(self, path: str) -> Dict:
        """从 `path` 目录恢复模型权重和训练进度。"""
        progress = self.trainer.load(path)
        self.learning_curve = progress.get('learning_curve', [])
        print(f"[{self.name}] 模型已从 {path} 加载")
        return progress


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────
def create_mappo_no_dod(config, env) -> MAPPOPolicy:
    """
    创建禁用 DoD 惩罚的 MAPPO 变体（消融实验用）。

    同时将 env 的 lyapunov_calc 替换为同一个无 DoD 计算器，
    确保训练期间奖励函数一致。

    Returns
    -------
    MAPPOPolicy（name='MAPPO_NoDod'）
    """
    lyapunov_no_dod = LyapunovCalculator(
        config, use_battery_loss=False, use_dod_penalty=False)
    env.lyapunov_calc = lyapunov_no_dod
    return MAPPOPolicy(config=config, lyapunov_calc=lyapunov_no_dod, name='MAPPO_NoDod')
