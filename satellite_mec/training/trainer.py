"""
training/trainer.py
===================
MAPPOTrainer：封装 PPO 更新、模型保存/加载。

负责
----
    - 维护 Actor / Critic / 优化器
    - compute_bootstrap()：从环境获取 V(s_T)
    - update()：执行一次 PPO 多轮 mini-batch 更新
    - save() / load()：持久化检查点

依赖
----
    training.networks  → Actor, Critic, RolloutBuffer
    core.env           → SatelliteMECEnv（类型标注用）
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import Actor, Critic, RolloutBuffer


class MAPPOTrainer:
    """
    MAPPO 训练器（共享参数，集中式 Critic）。

    使用示例
    --------
    >>> trainer = MAPPOTrainer(cfg)
    >>> bootstrap = trainer.compute_bootstrap(env, buffer)
    >>> metrics   = trainer.update(buffer, bootstrap)
    >>> trainer.save('./checkpoints/MAPPO')
    >>> trainer.load('./checkpoints/MAPPO')
    """

    def __init__(self, config):
        self.cfg    = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] 使用设备：{self.device}")

        self.actor     = Actor(config).to(self.device)
        self.critic    = Critic(config).to(self.device)
        self.actor_old = Actor(config).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_old.eval()

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=config.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

        self.update_count:           int   = 0
        self.best_completion_rate:   float = 0.0
        self.train_logs:             List[Dict] = []

    # ── 训练接口 ──────────────────────────────────────────────────────────────

    def compute_bootstrap(self, env, buffer: RolloutBuffer) -> Dict[int, float]:
        """计算当前 env 状态下各卫星的 bootstrap 价值 V(s_T)。"""
        bootstrap = {}
        for sat_id, critic_state in env.get_critic_obs().items():
            state_t = torch.FloatTensor(critic_state).to(self.device)
            with torch.no_grad():
                value = self.critic.get_value(state_t)
            bootstrap[sat_id] = float(value.item())
        return bootstrap

    def update(self, buffer: RolloutBuffer, bootstrap_values: Dict[int, float]) -> Dict:
        """
        执行一次完整的 PPO 更新（含 GAE 计算和多轮 mini-batch）。

        Parameters
        ----------
        buffer           : 轨迹缓冲区，必须已 add_slot / add_task 完毕
        bootstrap_values : {sat_id: V(s_T)}

        Returns
        -------
        训练指标字典（空缓冲区时返回 {}）
        """
        cfg = self.cfg
        buffer.compute_gae(bootstrap_values, gamma=cfg.GAMMA, lambda_gae=cfg.LAMBDA_GAE)
        ab = buffer.get_actor_batch()
        cb = buffer.get_critic_batch()
        if not ab or not cb:
            return {}

        states        = torch.tensor(np.array(ab['states'],        dtype=np.float32)).to(self.device)
        actions       = torch.tensor(np.array(ab['actions'],        dtype=np.int64)).to(self.device)
        log_probs_old = torch.tensor(np.array(ab['log_probs_old'],  dtype=np.float32)).to(self.device)
        masks         = torch.tensor(np.array(ab['masks'],          dtype=np.float32)).to(self.device)
        advantages    = torch.tensor(np.array(ab['advantages'],     dtype=np.float32)).to(self.device)
        critic_states  = torch.tensor(np.array(cb['critic_states'],  dtype=np.float32)).to(self.device)
        target_returns = torch.tensor(np.array(cb['target_returns'], dtype=np.float32)).to(self.device)

        all_critic_losses, all_actor_losses = [], []
        all_ratios, all_entropies = [], []
        n_task, n_slot = len(states), len(critic_states)

        for _ in range(cfg.EPOCH):
            task_idx = np.random.permutation(n_task)
            slot_idx = np.random.permutation(n_slot)

            # Critic 更新
            for start in range(0, n_slot, cfg.MINIBATCH):
                idx    = slot_idx[start:min(start + cfg.MINIBATCH, n_slot)]
                values = self.critic.get_value(critic_states[idx])
                critic_loss = nn.MSELoss()(values, target_returns[idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                all_critic_losses.append(critic_loss.item())

            # Actor 更新（Clipped PPO）
            for start in range(0, n_task, cfg.MINIBATCH):
                idx = task_idx[start:min(start + cfg.MINIBATCH, n_task)]
                log_probs_new, entropies = self.actor.evaluate_action(
                    states[idx], masks[idx], actions[idx])
                ratios = torch.exp(log_probs_new - log_probs_old[idx])
                surr1  = ratios * advantages[idx]
                surr2  = torch.clamp(ratios, 1 - cfg.EPSILON, 1 + cfg.EPSILON) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - cfg.BETA * entropies.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                all_actor_losses.append(actor_loss.item())
                all_ratios.append(ratios.mean().item())
                all_entropies.append(entropies.mean().item())

        # 同步 old actor
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.update_count += 1

        metrics = {
            'update_count':    self.update_count,
            'critic_loss':     float(np.mean(all_critic_losses)),
            'actor_loss':      float(np.mean(all_actor_losses)),
            'ratio_mean':      float(np.mean(all_ratios)),
            'entropy_mean':    float(np.mean(all_entropies)),
            'buffer_task_size': buffer.size(),
            'buffer_slot_size': buffer.slot_size(),
        }
        self.train_logs.append(metrics)
        return metrics

    # ── 检查点 ────────────────────────────────────────────────────────────────

    def save(self, path: str, extra_info: Dict | None = None) -> None:
        """
        保存所有模型权重和优化器状态到 `path` 目录。

        保存内容
        --------
            actor.pth, critic.pth, actor_old.pth,
            actor_optimizer.pth, critic_optimizer.pth,
            progress.pth（含 update_count / best_completion_rate）
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),            os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(),           os.path.join(path, 'critic.pth'))
        torch.save(self.actor_old.state_dict(),        os.path.join(path, 'actor_old.pth'))
        torch.save(self.actor_optimizer.state_dict(),  os.path.join(path, 'actor_optimizer.pth'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, 'critic_optimizer.pth'))
        progress = {
            'update_count':          self.update_count,
            'best_completion_rate':  self.best_completion_rate,
        }
        if extra_info:
            progress.update(extra_info)
        torch.save(progress, os.path.join(path, 'progress.pth'))

    def load(self, path: str) -> Dict:
        """
        从 `path` 目录恢复所有权重和优化器状态。

        Returns
        -------
        progress 字典（含 update_count / best_completion_rate 等）
        """
        dev = self.device
        self.actor.load_state_dict(
            torch.load(os.path.join(path, 'actor.pth'), map_location=dev))
        self.critic.load_state_dict(
            torch.load(os.path.join(path, 'critic.pth'), map_location=dev))
        self.actor_old.load_state_dict(
            torch.load(os.path.join(path, 'actor_old.pth'), map_location=dev))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(path, 'actor_optimizer.pth'), map_location=dev))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(path, 'critic_optimizer.pth'), map_location=dev))
        progress = torch.load(os.path.join(path, 'progress.pth'), map_location=dev)
        self.update_count          = progress.get('update_count', 0)
        self.best_completion_rate  = progress.get('best_completion_rate', 0.0)
        return progress

    def print_metrics(self, metrics: Dict) -> None:
        print(
            f"[Update {metrics['update_count']:4d}] "
            f"Critic Loss: {metrics['critic_loss']:.4f} | "
            f"Actor Loss:  {metrics['actor_loss']:.4f} | "
            f"Ratio:       {metrics['ratio_mean']:.3f} | "
            f"Entropy:     {metrics['entropy_mean']:.3f} | "
            f"Buffer:      {metrics['buffer_task_size']} tasks"
        )
