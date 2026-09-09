import json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_AGENTS = 25
OBS_DIM = 54
GLOBAL_DIM = 245
ACTION_DIM = 5
HIDDEN = 64
BATCH = 32


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),
        )

    def logits(self, obs, mask):
        logits = self.net(obs)
        return logits.masked_fill(mask <= 0, -1e9)


class Value(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QCritic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AgentQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, ACTION_DIM)
        )

    def forward(self, obs):
        return self.net(obs)


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hyper_w = nn.Sequential(
            nn.Linear(GLOBAL_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N_AGENTS)
        )
        self.hyper_b = nn.Linear(GLOBAL_DIM, 1)

    def forward(self, agent_q, global_state):
        weights = F.softplus(self.hyper_w(global_state))
        bias = self.hyper_b(global_state).squeeze(-1)
        return (weights * agent_q).sum(-1) + bias


def make_batch():
    obs = torch.randn(BATCH, N_AGENTS, OBS_DIM)
    global_state = torch.randn(BATCH, GLOBAL_DIM)
    mask = (torch.rand(BATCH, N_AGENTS, ACTION_DIM) > 0.2).float()
    mask[..., 0] = 1.0
    shared_reward = torch.randn(BATCH)
    return obs, global_state, mask, shared_reward


def mappo_step(obs, global_state, mask, reward):
    actor = Actor()
    critic = Value(GLOBAL_DIM)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-3
    )
    flat_obs = obs.reshape(-1, OBS_DIM)
    flat_mask = mask.reshape(-1, ACTION_DIM)
    dist = Categorical(logits=actor.logits(flat_obs, flat_mask))
    action = dist.sample()
    value = critic(global_state)
    advantage = (reward - value).detach().repeat_interleave(N_AGENTS)
    loss = -(dist.log_prob(action) * advantage).mean()
    loss += 0.5 * F.mse_loss(value, reward)
    loss -= 0.01 * dist.entropy().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item(), action, flat_mask


def ippo_step(obs, global_state, mask, reward):
    del global_state
    actor = Actor()
    critic = Value(OBS_DIM)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-3
    )
    flat_obs = obs.reshape(-1, OBS_DIM)
    flat_mask = mask.reshape(-1, ACTION_DIM)
    dist = Categorical(logits=actor.logits(flat_obs, flat_mask))
    action = dist.sample()
    value = critic(flat_obs)
    local_reward = reward.repeat_interleave(N_AGENTS)
    advantage = (local_reward - value).detach()
    loss = -(dist.log_prob(action) * advantage).mean()
    loss += 0.5 * F.mse_loss(value, local_reward)
    loss -= 0.01 * dist.entropy().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item(), action, flat_mask


def maddpg_step(obs, global_state, mask, reward):
    actor = Actor()
    critic = QCritic(GLOBAL_DIM + N_AGENTS * ACTION_DIM)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    flat_obs = obs.reshape(-1, OBS_DIM)
    flat_mask = mask.reshape(-1, ACTION_DIM)
    logits = actor.logits(flat_obs, flat_mask)
    soft_actions = F.gumbel_softmax(logits, tau=1.0, hard=False)
    joint_action = soft_actions.reshape(BATCH, N_AGENTS * ACTION_DIM)
    q_value = critic(torch.cat([global_state, joint_action], -1))
    critic_loss = F.mse_loss(q_value, reward)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    for param in critic.parameters():
        param.requires_grad_(False)
    logits = actor.logits(flat_obs, flat_mask)
    soft_actions = F.gumbel_softmax(logits, tau=1.0, hard=False)
    joint_action = soft_actions.reshape(BATCH, N_AGENTS * ACTION_DIM)
    actor_loss = -critic(torch.cat([global_state, joint_action], -1)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    for param in critic.parameters():
        param.requires_grad_(True)

    action = soft_actions.argmax(-1)
    return (critic_loss + actor_loss).detach().item(), action, flat_mask


def qmix_step(obs, global_state, mask, reward):
    agent_q = AgentQ()
    mixer = Mixer()
    optimizer = torch.optim.Adam(
        list(agent_q.parameters()) + list(mixer.parameters()), lr=1e-3
    )
    q_values = agent_q(obs).masked_fill(mask <= 0, -1e9)
    actions = q_values.argmax(-1)
    chosen_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    q_total = mixer(chosen_q, global_state)
    loss = F.mse_loss(q_total, reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item(), actions.reshape(-1), mask.reshape(-1, ACTION_DIM)


def run_smoke_test():
    obs, global_state, mask, reward = make_batch()
    results = {}
    for name, fn in [
        ("MAPPO", mappo_step),
        ("IPPO", ippo_step),
        ("MADDPG", maddpg_step),
        ("QMIX", qmix_step),
    ]:
        start = time.time()
        loss, actions, flat_mask = fn(obs, global_state, mask, reward)
        valid = flat_mask[torch.arange(flat_mask.size(0)), actions] > 0
        assert math.isfinite(loss), f"{name}: non-finite loss"
        assert bool(valid.all()), f"{name}: selected masked action"
        results[name] = {
            "loss": loss,
            "all_actions_valid": True,
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        }
    results["shared_interface"] = {
        "n_agents": N_AGENTS,
        "obs_dim": OBS_DIM,
        "global_dim": GLOBAL_DIM,
        "action_dim": ACTION_DIM,
        "shared_reward": True,
        "dynamic_action_mask": True,
        "seed": SEED,
    }
    return results


if __name__ == "__main__":
    result = run_smoke_test()
    print(json.dumps(result, indent=2))
    os.makedirs("results/backbone_compare/smoke", exist_ok=True)
    with open("results/backbone_compare/smoke/smoke_results.json", "w") as f:
        json.dump(result, f, indent=2)
