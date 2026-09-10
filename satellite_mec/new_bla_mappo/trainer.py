"""PPO trainer for paper-faithful new BLA-MAPPO."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.networks import Actor, Critic


class NewBLAMAPPOTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)
        self.update_count = 0

    def value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return float(self.critic.get_value(tensor).item())

    def update(self, buffer, bootstrap_values) -> dict:
        buffer.compute_advantages(bootstrap_values)
        actor_batch = buffer.actor_batch()
        critic_batch = buffer.critic_batch()
        if not actor_batch or not critic_batch:
            return {}

        states = torch.as_tensor(actor_batch["states"], device=self.device)
        actions = torch.as_tensor(actor_batch["actions"], device=self.device)
        old_log_probs = torch.as_tensor(actor_batch["log_probs_old"], device=self.device)
        masks = torch.as_tensor(actor_batch["masks"], device=self.device)
        advantages = torch.as_tensor(actor_batch["advantages"], device=self.device)
        critic_states = torch.as_tensor(critic_batch["states"], device=self.device)
        returns = torch.as_tensor(critic_batch["returns"], device=self.device)

        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.cfg.EPOCH):
            critic_order = np.random.permutation(len(critic_states))
            for start in range(0, len(critic_states), self.cfg.MINIBATCH):
                idx = critic_order[start:start + self.cfg.MINIBATCH]
                values = self.critic.get_value(critic_states[idx])
                loss = nn.functional.mse_loss(values, returns[idx])
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                critic_losses.append(float(loss.item()))

            actor_order = np.random.permutation(len(states))
            for start in range(0, len(states), self.cfg.MINIBATCH):
                idx = actor_order[start:start + self.cfg.MINIBATCH]
                log_probs, entropy = self.actor.evaluate_action(states[idx], masks[idx], actions[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.EPSILON, 1.0 + self.cfg.EPSILON
                ) * advantages[idx]
                # Maximize Eq. (38), implemented as a minimization loss.
                loss = -torch.minimum(unclipped, clipped).mean() - self.cfg.BETA * entropy.mean()
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                actor_losses.append(float(loss.item()))
                entropies.append(float(entropy.mean().item()))

        self.update_count += 1
        return {
            "update_count": self.update_count,
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "actor_samples": len(states),
            "critic_samples": len(critic_states),
        }

    def save(self, path: str, extra: dict | None = None) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, "actor_optimizer.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, "critic_optimizer.pth"))
        torch.save({"update_count": self.update_count, **(extra or {})},
                   os.path.join(path, "progress.pth"))

    def load(self, path: str) -> dict:
        dev = self.device
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=dev))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), map_location=dev))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(path, "actor_optimizer.pth"), map_location=dev))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(path, "critic_optimizer.pth"), map_location=dev))
        progress = torch.load(os.path.join(path, "progress.pth"), map_location=dev)
        self.update_count = int(progress.get("update_count", 0))
        return progress
