#!/usr/bin/env python3
"""Train MAPPO/IPPO/MADDPG/TD3 under one fixed reward/evaluation protocol.

The script is intentionally self-contained so debug, pilot and 32K runs use the
same code path.  Environment slots -- never replay insertions or PPO episodes --
are the canonical training x-axis.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from baselines.maddpg_dod import MADDPGDoDPolicy
from baselines.td3_sched import TD3SchedPolicy
from core.common_reward import (
    COMMON_REWARD_V1,
    REWARD_COMPONENTS,
    REWARD_DEFINITION,
    CommonRewardEnv,
    assert_legacy_shaping_disabled,
)
from core.config import Config
from core.env import SatelliteMECEnv
from training.ippo import IPPOPolicy, LOCAL_CRITIC_DIM
from training.policy import MAPPOPolicy


ALGORITHMS = ("MAPPO", "IPPO", "MADDPG", "TD3")
EVAL_SEEDS = {"task": 424242, "task_param": 424243, "dod_init": 424244}


class ComparisonConfig(Config):
    """Config evaluated by ``Config.__init__`` with experiment constants active."""

    LAMBDA_HIGH = 4.0
    LAMBDA = LAMBDA_HIGH * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    V = 50.0
    ETA = 0.5
    BETA = 0.02
    BETA_TASK = 0.5
    K_ROLLOUT = 64
    T_TRAIN = 32000
    T_EVAL = 500
    T_TOTAL = 40000

    # The base env must produce *only* Lyapunov action cost. common_reward_v1 is
    # applied exactly once by CommonRewardEnv in both train and eval.
    COMPLETION_BONUS = 0.0
    W_DONE = 0.0
    W_TIMEOUT = 0.0
    W_REJECT = 0.0
    W_HL = 0.0
    W_DOD = 0.0
    W_QUEUE = 0.0


def make_config(max_steps: int, eval_slots: int) -> ComparisonConfig:
    cfg = ComparisonConfig()
    cfg.T_TRAIN = int(max_steps)
    cfg.T_EVAL = int(eval_slots)
    # solar_seq is built when the base env is constructed; the class-level 40K
    # horizon covers the required 32K experiment and its 500-slot evaluations.
    if max_steps > ComparisonConfig.T_TOTAL:
        raise ValueError(f"steps={max_steps} exceeds supported solar horizon {ComparisonConfig.T_TOTAL}")
    assert_legacy_shaping_disabled(cfg)
    return cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


class _PPOUpdateCounterMixin:
    actor_gradient_updates: int
    critic_gradient_updates: int

    def _init_update_counters(self) -> None:
        self.actor_gradient_updates = 0
        self.critic_gradient_updates = 0

    def _trigger_update(self, env):
        n_task = self.buffer.size()
        n_slot = self.buffer.slot_size()
        if n_task == 0:
            return None
        actor_steps = self.cfg.EPOCH * math.ceil(n_task / self.cfg.MINIBATCH)
        critic_steps = self.cfg.EPOCH * math.ceil(n_slot / self.cfg.MINIBATCH)
        metrics = super()._trigger_update(env)
        if metrics:
            self.actor_gradient_updates += int(actor_steps)
            self.critic_gradient_updates += int(critic_steps)
        return metrics

    @property
    def gradient_updates(self) -> int:
        return int(self.actor_gradient_updates + self.critic_gradient_updates)


class CountedMAPPO(_PPOUpdateCounterMixin, MAPPOPolicy):
    def __init__(self, config):
        super().__init__(config, name="MAPPO")
        self._init_update_counters()


class CountedIPPO(_PPOUpdateCounterMixin, IPPOPolicy):
    def __init__(self, config):
        super().__init__(config, name="IPPO")
        self._init_update_counters()


class CountedMADDPG(MADDPGDoDPolicy):
    """MADDPG with true optimizer counters and replay action audit metadata."""

    def __init__(self, *args, **kwargs):
        kwargs["zhong_reward"] = False
        kwargs["reward_scale"] = 1.0
        super().__init__(*args, **kwargs)
        self.actor_gradient_updates = 0
        self.critic_gradient_updates = 0
        cap = self.replay.capacity
        self.replay.network_action = np.zeros((cap, self.a_dim), dtype=np.float32)
        self.replay.executed_action = np.zeros((cap,), dtype=np.int16)
        self.replay.executed_mask = np.zeros((cap, self.a_dim), dtype=np.float32)
        self._last_network_action = np.zeros(self.a_dim, dtype=np.float32)
        self._slot_audit: List[Tuple[np.ndarray, int, np.ndarray]] = []
        self._audit_pending = None

    @property
    def gradient_updates(self) -> int:
        return int(self.actor_gradient_updates + self.critic_gradient_updates)

    def act_one(self, state: np.ndarray, mask: np.ndarray):
        s_t = torch.as_tensor(np.asarray(state, np.float32), device=self.device).unsqueeze(0)
        m_t = torch.as_tensor(np.asarray(mask, np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_logits = self.actor(s_t)
            self._last_network_action = raw_logits.squeeze(0).cpu().numpy().astype(np.float32)
            logits = raw_logits
            if not self._eval_mode and self.expl_noise > 0:
                logits = logits + torch.randn_like(logits) * self.expl_noise
            if self._eval_mode:
                action = int(logits.masked_fill(m_t < 0.5, -1e9).argmax(dim=-1).item())
            else:
                action = int(self._gumbel_softmax(logits, m_t, self._gs_temp(), hard=True)
                             .argmax(dim=-1).item())
        return action, 0.0

    def record_task_transition(self, sat_id, slot_t, state, action, log_prob, mask, task_reward):
        before = len(self._slot_tasks)
        super().record_task_transition(sat_id, slot_t, state, action, log_prob, mask, task_reward)
        if len(self._slot_tasks) > before:
            self._slot_audit.append((self._last_network_action.copy(), int(action),
                                     np.asarray(mask, np.float32).copy()))

    def _write_audit(self, index: int, audit) -> None:
        net, executed, mask = audit
        self.replay.network_action[index] = net
        self.replay.executed_action[index] = executed
        self.replay.executed_mask[index] = mask

    def _flush_slot(self, rewards: Dict, done: bool, info: Optional[Dict] = None) -> None:
        if self.zhong_reward:
            raise RuntimeError("common_reward_v1 experiment forbids Zhong-specific reward")
        if not self._slot_tasks:
            self._slot_audit = []
            return
        count: Dict[int, int] = {}
        sum_cost: Dict[int, float] = {}
        for (_, _, _, _, ac, n) in self._slot_tasks:
            count[n] = count.get(n, 0) + 1
            sum_cost[n] = sum_cost.get(n, 0.0) + float(ac)
        outcome = {n: float(rewards.get(n, 0.0)) - sum_cost[n] for n in count}
        for task, audit in zip(self._slot_tasks, self._slot_audit):
            s, cobs, a, mask, ac, n = task
            r = float(ac + outcome[n] / max(count[n], 1))
            if self._pending is not None:
                ps, pc, pa, pr = self._pending
                i = self.replay.ptr
                self.replay.add(ps, pc, pa, pr, s, cobs, mask, 0.0)
                self._write_audit(i, self._audit_pending)
                self._env_steps += 1
            self._pending = (s, cobs, a, r)
            self._audit_pending = audit
        self._slot_tasks = []
        self._slot_audit = []
        if done and self._pending is not None:
            ps, pc, pa, pr = self._pending
            i = self.replay.ptr
            self.replay.add(ps, pc, pa, pr, np.zeros(self.s_dim, np.float32),
                            np.zeros(self.c_dim, np.float32), np.ones(self.a_dim, np.float32), 1.0)
            self._write_audit(i, self._audit_pending)
            self._env_steps += 1
            self._pending = None
            self._audit_pending = None

    def _update(self) -> None:
        eligible = self.replay.size >= max(self.batch_size, self.start_steps)
        super()._update()
        if eligible:
            # Existing MADDPG implementation performs one critic and one actor
            # optimizer step on every eligible update call.
            self.critic_gradient_updates += 1
            self.actor_gradient_updates += 1


class CountedTD3(TD3SchedPolicy):
    """TD3 with explicit network/executed action records and update counters."""

    def __init__(self, *args, **kwargs):
        kwargs["reward_scale"] = 1.0
        super().__init__(*args, **kwargs)
        self.actor_gradient_updates = 0
        self.critic_gradient_updates = 0
        cap = self.replay.capacity
        self.replay.network_action = np.zeros((cap, self.action_dim), dtype=np.float32)
        self.replay.executed_action = np.zeros((cap,), dtype=np.int16)
        self.replay.executed_mask = np.zeros((cap, self.action_dim), dtype=np.float32)
        self._slot_audit: List[Tuple[np.ndarray, int, np.ndarray]] = []
        self._audit_pending = None

    @property
    def gradient_updates(self) -> int:
        return int(self.actor_gradient_updates + self.critic_gradient_updates)

    def record_task_transition(self, sat_id, slot_t, state, action, log_prob, mask, task_reward):
        before = len(self._slot_tasks)
        super().record_task_transition(sat_id, slot_t, state, action, log_prob, mask, task_reward)
        if len(self._slot_tasks) > before:
            self._slot_audit.append((self._last_a_cont.copy(), int(action),
                                     np.asarray(mask, np.float32).copy()))

    def _write_audit(self, index: int, audit) -> None:
        net, executed, mask = audit
        self.replay.network_action[index] = net
        self.replay.executed_action[index] = executed
        self.replay.executed_mask[index] = mask

    def _flush_slot(self, rewards: Dict, done: bool) -> None:
        if not self._slot_tasks:
            self._slot_audit = []
            return
        sum_cost: Dict[int, float] = {}
        count: Dict[int, int] = {}
        for (_, _, ac, n) in self._slot_tasks:
            sum_cost[n] = sum_cost.get(n, 0.0) + float(ac)
            count[n] = count.get(n, 0) + 1
        outcome = {n: float(rewards.get(n, 0.0)) - sum_cost[n] for n in count}
        for task, audit in zip(self._slot_tasks, self._slot_audit):
            s, a_cont, ac, n = task
            r = float(ac + outcome[n] / max(count[n], 1))
            if self._pending is not None:
                ps, pa, pr = self._pending
                i = self.replay.ptr
                self.replay.add(ps, pa, pr, s, 0.0)
                self._write_audit(i, self._audit_pending)
                self._env_steps += 1
            self._pending = (s, a_cont, r)
            self._audit_pending = audit
        self._slot_tasks = []
        self._slot_audit = []
        if done and self._pending is not None:
            ps, pa, pr = self._pending
            i = self.replay.ptr
            self.replay.add(ps, pa, pr, np.zeros(self.state_dim, np.float32), 1.0)
            self._write_audit(i, self._audit_pending)
            self._env_steps += 1
            self._pending = None
            self._audit_pending = None

    def _update(self) -> None:
        before = int(self._total_it)
        super()._update()
        if self._total_it > before:
            self.critic_gradient_updates += 1
            if self._total_it % self.policy_freq == 0:
                self.actor_gradient_updates += 1


def build_env(cfg: ComparisonConfig, phase: str, seeds: Optional[Dict[str, int]] = None) -> CommonRewardEnv:
    env = CommonRewardEnv(SatelliteMECEnv(cfg), spec=COMMON_REWARD_V1)
    env.reset(phase=phase, seeds=seeds)
    return env


def build_policy(algorithm: str, cfg: ComparisonConfig, train_env: CommonRewardEnv):
    set_all_seeds(cfg.SEED)
    if algorithm == "MAPPO":
        return CountedMAPPO(cfg)
    if algorithm == "IPPO":
        return CountedIPPO(cfg)
    if algorithm == "MADDPG":
        return CountedMADDPG(cfg, train_env.base_env, seed=cfg.SEED,
                             zhong_reward=False, reward_scale=1.0)
    if algorithm == "TD3":
        return CountedTD3(cfg, train_env.base_env, seed=cfg.SEED, reward_scale=1.0)
    raise ValueError(algorithm)


class DeterministicActorAdapter:
    """Read-only actor adapter: no training hooks exist on this object."""

    def __init__(self, algorithm: str, policy):
        self.algorithm = algorithm
        self.policy = policy

    def act_one(self, state: np.ndarray, mask: np.ndarray):
        mask_np = np.asarray(mask, dtype=np.float32)
        if self.algorithm in ("MAPPO", "IPPO"):
            actor = self.policy.actor
            device = self.policy.trainer.device
            s = torch.as_tensor(state, dtype=torch.float32, device=device)
            m = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
            action, log_prob, _ = actor.get_action(s, m, deterministic=True)
            return action, log_prob
        s = torch.as_tensor(np.asarray(state, np.float32), dtype=torch.float32,
                            device=self.policy.device).unsqueeze(0)
        with torch.no_grad():
            values = self.policy.actor(s).squeeze(0).cpu().numpy()
        action = int(np.argmax(np.where(mask_np > 0, values, -1e30)))
        return action, 0.0


def _tensor_digest(modules: Iterable[torch.nn.Module]) -> str:
    h = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            h.update(name.encode())
            a = tensor.detach().cpu().contiguous().numpy()
            h.update(a.tobytes())
    return h.hexdigest()


def policy_fingerprint(algorithm: str, policy) -> Dict[str, object]:
    if algorithm in ("MAPPO", "IPPO"):
        modules = [policy.actor, policy.critic, policy.trainer.actor_old]
        return {
            "model": _tensor_digest(modules),
            "gradient_updates": policy.gradient_updates,
            "actor_updates": policy.actor_gradient_updates,
            "critic_updates": policy.critic_gradient_updates,
            "ppo_update_count": policy.trainer.update_count,
            "buffer_tasks": policy.buffer.size(),
            "buffer_slots": policy.buffer.slot_size(),
        }
    modules = [policy.actor, policy.actor_target, policy.critic, policy.critic_target]
    fp = {
        "model": _tensor_digest(modules),
        "gradient_updates": policy.gradient_updates,
        "actor_updates": policy.actor_gradient_updates,
        "critic_updates": policy.critic_gradient_updates,
        "replay_size": int(policy.replay.size),
        "replay_ptr": int(policy.replay.ptr),
    }
    if policy.replay.size:
        # Evaluation has no method capable of mutating replay.  This digest still
        # covers all replay rows that encode reward and both network/executed action.
        h = hashlib.sha256()
        n = int(policy.replay.size)
        for field in ("r", "network_action", "executed_action", "executed_mask"):
            h.update(np.ascontiguousarray(getattr(policy.replay, field)[:n]).tobytes())
        fp["replay_content"] = h.hexdigest()
    else:
        fp["replay_content"] = hashlib.sha256(b"").hexdigest()
    if algorithm == "TD3":
        fp["td3_critic_iterations"] = int(policy._total_it)
    return fp


def evaluate_fixed(algorithm: str, policy, cfg: ComparisonConfig, slots: int) -> Dict[str, object]:
    before = policy_fingerprint(algorithm, policy)
    eval_cfg = make_config(cfg.T_TRAIN, slots)
    eval_env = build_env(eval_cfg, "eval", EVAL_SEEDS)
    adapter = DeterministicActorAdapter(algorithm, policy)

    totals = {k: 0.0 for k in (*REWARD_COMPONENTS, "total")}
    arrived = done_count = satisfied = timeout = 0
    delays: List[float] = []
    health: List[float] = []
    queue_tasks: List[float] = []
    for _ in range(int(slots)):
        _, rewards, _, info = eval_env.step(policy=adapter)
        ledger = info["reward_ledger"]
        if not math.isclose(sum(rewards.values()), ledger["total"], rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError("evaluation rewards do not sum to common ledger total")
        for k in totals:
            totals[k] += float(ledger[k])
        arrived += int(info.get("arrived", 0))
        done_count += int(info.get("done_tasks", 0))
        satisfied += int(info.get("slot_satisfied", 0))
        timeout += int(info.get("slot_timeout", 0))
        delays.extend(float(x) for x in info.get("slot_e2e_delays", []))
        health.append(float(info.get("avg_health_loss", 0.0)))
        queue_tasks.append(float(info.get("queue_task_count", 0.0)))

    after = policy_fingerprint(algorithm, policy)
    if before != after:
        raise AssertionError(f"fixed evaluation mutated {algorithm} training state")
    return {
        "eval_return_total": float(totals["total"]),
        "eval_return_per_slot": float(totals["total"] / max(slots, 1)),
        "completion_rate": float(done_count / max(arrived, 1)),
        "satisfaction": float(satisfied / max(done_count + timeout, 1)),
        "avg_delay": float(np.mean(delays)) if delays else 0.0,
        "avg_health_loss": float(np.mean(health)) if health else 0.0,
        "queue_tasks_per_sat": float(np.mean(queue_tasks)) if queue_tasks else 0.0,
        "reward_ledger": {k: float(v / max(slots, 1)) for k, v in totals.items()},
        "eval_arrived": int(arrived),
        "eval_done": int(done_count),
        "eval_timeout": int(timeout),
    }


def new_history(algorithm: str, seed: int) -> Dict[str, object]:
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "reward_definition": REWARD_DEFINITION,
        "environment_steps": [],
        "gradient_updates": [],
        "actor_gradient_updates": [],
        "critic_gradient_updates": [],
        "wall_time_seconds": [],
        "train_return": [],
        "train_return_definition": "mean common system reward per environment slot since previous evaluation",
        "train_return_interval_mean": [],
        "eval_return": [],
        "eval_return_total": [],
        "eval_return_per_slot": [],
        "completion_rate": [],
        "satisfaction": [],
        "avg_delay": [],
        "avg_health_loss": [],
        "queue_tasks_per_sat": [],
        "reward_ledger": {k: [] for k in (*REWARD_COMPONENTS, "total")},
    }


def append_history(history: Dict[str, object], step: int, wall: float, train_mean: float,
                   policy, metrics: Mapping[str, object]) -> None:
    h = history
    h["environment_steps"].append(int(step))
    h["gradient_updates"].append(int(policy.gradient_updates))
    h["actor_gradient_updates"].append(int(policy.actor_gradient_updates))
    h["critic_gradient_updates"].append(int(policy.critic_gradient_updates))
    h["wall_time_seconds"].append(float(wall))
    h["train_return"].append(float(train_mean))
    h["train_return_interval_mean"].append(float(train_mean))
    h["eval_return"].append(float(metrics["eval_return_total"]))
    h["eval_return_total"].append(float(metrics["eval_return_total"]))
    h["eval_return_per_slot"].append(float(metrics["eval_return_per_slot"]))
    for k in ("completion_rate", "satisfaction", "avg_delay", "avg_health_loss", "queue_tasks_per_sat"):
        h[k].append(float(metrics[k]))
    for k in (*REWARD_COMPONENTS, "total"):
        h["reward_ledger"][k].append(float(metrics["reward_ledger"][k]))
    validate_history(h)


def validate_history(history: Mapping[str, object]) -> None:
    n = len(history["environment_steps"])
    vector_keys = (
        "gradient_updates", "actor_gradient_updates", "critic_gradient_updates",
        "wall_time_seconds", "train_return", "train_return_interval_mean",
        "eval_return", "eval_return_total", "eval_return_per_slot", "completion_rate",
        "satisfaction", "avg_delay", "avg_health_loss", "queue_tasks_per_sat",
    )
    for key in vector_keys:
        if len(history[key]) != n:
            raise AssertionError(f"history length mismatch: {key}")
        if not np.all(np.isfinite(np.asarray(history[key], dtype=float))):
            raise AssertionError(f"history contains NaN/Inf: {key}")
    for key, values in history["reward_ledger"].items():
        if len(values) != n or not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise AssertionError(f"invalid reward ledger series: {key}")


def _replay_state(policy, algorithm: str) -> Optional[Dict[str, object]]:
    if algorithm not in ("MADDPG", "TD3"):
        return None
    r = policy.replay
    n = int(r.size)
    fields = ["s", "a", "r", "s2", "d", "network_action", "executed_action", "executed_mask"]
    if algorithm == "MADDPG":
        fields += ["c", "c2", "m2"]
    return {
        "size": n, "ptr": int(r.ptr),
        "arrays": {f: np.asarray(getattr(r, f)[:n]).copy() for f in fields},
    }


def checkpoint_payload(algorithm: str, policy, train_env: CommonRewardEnv,
                       history: Mapping, step: int, wall: float) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "algorithm": algorithm,
        "step": int(step), "wall": float(wall), "history": copy.deepcopy(history),
        "env_pickle": pickle.dumps(train_env, protocol=pickle.HIGHEST_PROTOCOL),
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "actor": policy.actor.state_dict(),
        "actor_updates": int(policy.actor_gradient_updates),
        "critic_updates": int(policy.critic_gradient_updates),
    }
    if algorithm in ("MAPPO", "IPPO"):
        payload.update({
            "critic": policy.critic.state_dict(),
            "actor_old": policy.trainer.actor_old.state_dict(),
            "actor_optimizer": policy.trainer.actor_optimizer.state_dict(),
            "critic_optimizer": policy.trainer.critic_optimizer.state_dict(),
            "ppo_update_count": int(policy.trainer.update_count),
            "buffer_pickle": pickle.dumps(policy.buffer, protocol=pickle.HIGHEST_PROTOCOL),
        })
    else:
        payload.update({
            "actor_target": policy.actor_target.state_dict(),
            "critic": policy.critic.state_dict(),
            "critic_target": policy.critic_target.state_dict(),
            "actor_optimizer": policy.actor_opt.state_dict(),
            "critic_optimizer": policy.critic_opt.state_dict(),
            "replay": _replay_state(policy, algorithm),
            "pending": copy.deepcopy(policy._pending),
            "audit_pending": copy.deepcopy(policy._audit_pending),
            "transition_count": int(policy._env_steps),
        })
        if algorithm == "MADDPG":
            payload["Y"] = copy.deepcopy(policy._Y)
            payload["prevQ"] = copy.deepcopy(policy._prevQ)
        else:
            payload["total_it"] = int(policy._total_it)
    return payload


def save_checkpoint(path: Path, algorithm: str, policy, train_env, history, step, wall) -> str:
    path.mkdir(parents=True, exist_ok=True)
    target = path / "training_state.pt"
    tmp = path / "training_state.pt.tmp"
    torch.save(checkpoint_payload(algorithm, policy, train_env, history, step, wall), tmp)
    os.replace(tmp, target)
    h = hashlib.sha256(target.read_bytes()).hexdigest()
    (path / "SHA256.txt").write_text(f"{h}  training_state.pt\n", encoding="utf-8")
    return h


def load_checkpoint(path: Path, algorithm: str, cfg: ComparisonConfig):
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if payload["algorithm"] != algorithm:
        raise ValueError("checkpoint algorithm mismatch")
    train_env: CommonRewardEnv = pickle.loads(payload["env_pickle"])
    policy = build_policy(algorithm, cfg, train_env)
    policy.actor.load_state_dict(payload["actor"])
    policy.actor_gradient_updates = int(payload["actor_updates"])
    policy.critic_gradient_updates = int(payload["critic_updates"])
    if algorithm in ("MAPPO", "IPPO"):
        policy.critic.load_state_dict(payload["critic"])
        policy.trainer.actor_old.load_state_dict(payload["actor_old"])
        policy.trainer.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        policy.trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        policy.trainer.update_count = int(payload["ppo_update_count"])
        policy.buffer = pickle.loads(payload["buffer_pickle"])
    else:
        policy.actor_target.load_state_dict(payload["actor_target"])
        policy.critic.load_state_dict(payload["critic"])
        policy.critic_target.load_state_dict(payload["critic_target"])
        policy.actor_opt.load_state_dict(payload["actor_optimizer"])
        policy.critic_opt.load_state_dict(payload["critic_optimizer"])
        rstate = payload["replay"]
        policy.replay.size = int(rstate["size"]); policy.replay.ptr = int(rstate["ptr"])
        for field, arr in rstate["arrays"].items():
            getattr(policy.replay, field)[:len(arr)] = arr
        policy._pending = payload["pending"]
        policy._audit_pending = payload["audit_pending"]
        policy._env_steps = int(payload["transition_count"])
        if algorithm == "MADDPG":
            policy._Y = payload.get("Y", {}); policy._prevQ = payload.get("prevQ", {})
            policy.env = train_env.base_env
        else:
            policy._total_it = int(payload.get("total_it", 0)); policy.env = train_env.base_env
    np.random.set_state(payload["numpy_random_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    return policy, train_env, payload["history"], int(payload["step"]), float(payload["wall"])


def deterministic_signature(algorithm: str, policy, cfg: ComparisonConfig) -> Tuple[int, np.ndarray]:
    state = np.linspace(-0.5, 0.5, cfg.get_state_dim(), dtype=np.float32)
    mask = np.array([1, 1, 0, 1, 1], dtype=np.float32)
    adapter = DeterministicActorAdapter(algorithm, policy)
    action, _ = adapter.act_one(state, mask)
    s = torch.as_tensor(state, dtype=torch.float32)
    if algorithm in ("MAPPO", "IPPO"):
        device = policy.trainer.device
        with torch.no_grad():
            out = policy.actor(s.to(device), torch.as_tensor(mask, device=device)).cpu().numpy()
    else:
        with torch.no_grad():
            out = policy.actor(s.unsqueeze(0)).squeeze(0).cpu().numpy()
    return int(action), np.asarray(out)


def verify_checkpoint_roundtrip(algorithm: str, cfg: ComparisonConfig, ckpt: Path,
                                original_policy) -> None:
    sig_before = deterministic_signature(algorithm, original_policy, cfg)
    loaded_policy, _, _, _, _ = load_checkpoint(ckpt, algorithm, cfg)
    sig_after = deterministic_signature(algorithm, loaded_policy, cfg)
    if sig_before[0] != sig_after[0] or not np.array_equal(sig_before[1], sig_after[1]):
        raise AssertionError(f"{algorithm} checkpoint deterministic output mismatch")
    if original_policy.gradient_updates != loaded_policy.gradient_updates:
        raise AssertionError(f"{algorithm} checkpoint update-count mismatch")


def run_algorithm(algorithm: str, cfg: ComparisonConfig, out_dir: Path, steps: int,
                  eval_interval: int, eval_slots: int, resume: bool) -> Tuple[Dict, Dict]:
    algo_dir = out_dir / algorithm.lower()
    ckpt_dir = algo_dir / "checkpoint"
    hist_path = algo_dir / "history.json"
    if resume and (ckpt_dir / "training_state.pt").exists():
        policy, train_env, history, start_step, train_wall = load_checkpoint(ckpt_dir, algorithm, cfg)
    else:
        set_all_seeds(cfg.SEED)
        train_env = build_env(cfg, "train", {"task": cfg.SEED_TASK, "task_param": cfg.SEED_TASK_PARAM,
                                              "dod_init": cfg.SEED})
        policy = build_policy(algorithm, cfg, train_env)
        history = new_history(algorithm, cfg.SEED)
        start_step = 0
        train_wall = 0.0
        metrics = evaluate_fixed(algorithm, policy, cfg, eval_slots)
        append_history(history, 0, 0.0, 0.0, policy, metrics)
        atomic_json(hist_path, history)
        save_checkpoint(ckpt_dir, algorithm, policy, train_env, history, 0, 0.0)

    if start_step > steps:
        raise ValueError(f"checkpoint at {start_step} exceeds requested steps={steps}")
    interval_rewards: List[float] = []
    last_eval_step = int(history["environment_steps"][-1])
    for env_step in range(start_step + 1, steps + 1):
        t0 = time.perf_counter()
        rewards, _, info = policy.run_step(train_env)
        train_wall += time.perf_counter() - t0
        ledger_total = float(info["reward_ledger"]["total"])
        if not math.isclose(sum(rewards.values()), ledger_total, rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"{algorithm}: train rewards != common ledger")
        if not math.isfinite(ledger_total):
            raise FloatingPointError(f"{algorithm}: non-finite training reward")
        interval_rewards.append(ledger_total)

        is_eval = env_step % eval_interval == 0 or env_step == steps
        if is_eval:
            metrics = evaluate_fixed(algorithm, policy, cfg, eval_slots)
            mean_train = float(np.mean(interval_rewards)) if interval_rewards else 0.0
            append_history(history, env_step, train_wall, mean_train, policy, metrics)
            interval_rewards.clear()
            atomic_json(hist_path, history)
            checksum = save_checkpoint(ckpt_dir, algorithm, policy, train_env, history,
                                       env_step, train_wall)
            last_eval_step = env_step
            print(f"[{algorithm}] env_steps={env_step} grad={policy.gradient_updates} "
                  f"eval/slot={metrics['eval_return_per_slot']:.6f} "
                  f"CR={metrics['completion_rate']:.4f} checkpoint={checksum[:12]}", flush=True)

    if steps >= 1000 and policy.gradient_updates <= 0:
        raise AssertionError(f"{algorithm} completed {steps} environment slots with zero gradient updates")
    verify_checkpoint_roundtrip(algorithm, cfg, ckpt_dir, policy)
    return history, {
        "algorithm": algorithm,
        "environment_steps": int(last_eval_step),
        "gradient_updates": int(policy.gradient_updates),
        "actor_gradient_updates": int(policy.actor_gradient_updates),
        "critic_gradient_updates": int(policy.critic_gradient_updates),
        "wall_time_seconds": float(train_wall),
        "checkpoint": str(ckpt_dir / "training_state.pt"),
        "checkpoint_sha256": (ckpt_dir / "SHA256.txt").read_text().split()[0],
        "actor_input_dim": int(cfg.get_state_dim()),
        "actor_parameters": count_parameters(policy.actor),
        "critic_input_dim": int(LOCAL_CRITIC_DIM if algorithm == "IPPO" else
                                cfg.get_critic_state_dim() if algorithm in ("MAPPO", "MADDPG") else
                                cfg.get_state_dim()),
        "critic_parameters": count_parameters(policy.critic),
        "reward_scale": 1.0,
    }


def _git(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def package_versions() -> Dict[str, str]:
    import matplotlib
    import scipy
    import tqdm
    return {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "tqdm": tqdm.__version__,
    }


def output_tag(steps: int) -> str:
    if steps == 32000:
        return "seed_42"
    if steps == 8000:
        return "pilot_8000_seed_42"
    if steps == 1000:
        return "debug_1000_seed_42"
    if steps == 10:
        return "smoke_10_seed_42"
    return f"steps_{steps}_seed_42"


def write_comparison(out_dir: Path, histories: Mapping[str, Mapping]) -> None:
    rows = []
    for alg in ALGORITHMS:
        h = histories[alg]
        i = -1
        rows.append({
            "algorithm": alg,
            "environment_steps": h["environment_steps"][i],
            "gradient_updates": h["gradient_updates"][i],
            "wall_time_seconds": h["wall_time_seconds"][i],
            "eval_return_per_slot": h["eval_return_per_slot"][i],
            "completion_rate": h["completion_rate"][i],
            "satisfaction": h["satisfaction"][i],
            "avg_delay": h["avg_delay"][i],
            "avg_health_loss": h["avg_health_loss"][i],
            "queue_tasks_per_sat": h["queue_tasks_per_sat"][i],
        })
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def make_manifest(cfg: ComparisonConfig, args, out_dir: Path,
                  summaries: Mapping[str, Mapping]) -> Dict[str, object]:
    return {
        "preliminary": True,
        "single_seed": True,
        "seed": 42,
        "derived_seeds": {
            "train_task": cfg.SEED_TASK, "train_task_param": cfg.SEED_TASK_PARAM,
            "train_link": cfg.SEED_LINK, "train_network": cfg.SEED_NET,
            "train_optimizer": cfg.SEED_TRAIN, "eval": EVAL_SEEDS,
        },
        "reward_definition": COMMON_REWARD_V1.to_dict(),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"]),
            "status_short": _git(["status", "--short"]), "log_5": _git(["log", "-5", "--oneline"]),
            "remote": _git(["remote", "-v"]),
        },
        "versions": package_versions(),
        "hardware": {
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": {k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool))},
        "fixed_config": {
            "N_PLANES": 5, "N_SATS_PER_PLANE": 5, "N_SATS": 25,
            "K_ROLLOUT": 64, "LAMBDA_HIGH": 4.0, "V": 50.0, "ETA": 0.5,
            "BETA": 0.02, "BETA_TASK": 0.5,
        },
        "protocol": {
            "training_environment_steps": args.steps,
            "eval_interval_environment_steps": args.eval_interval,
            "eval_slots": args.eval_slots,
            "evaluation_isolation": "independent environment + read-only deterministic actor adapter",
            "main_x_axis": "environment_steps",
            "action_mapping": {"0": "local", "1": "neighbor_1", "2": "neighbor_2",
                               "3": "neighbor_3", "4": "neighbor_4"},
            "action_masking": "same Satellite.get_action_mask; illegal logits/preferences masked before argmax",
            "training_reward_scale": 1.0,
        },
        "algorithms": summaries,
        "command": " ".join([sys.executable, *sys.argv]),
        "resume_requested": bool(args.resume),
        "raw_results_directory": str(out_dir),
        "output_files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithms", default=",".join(ALGORITHMS),
                   help="comma-separated subset: MAPPO,IPPO,MADDPG,TD3")
    p.add_argument("--resume", action="store_true", help="resume exact saved training state when present")
    p.add_argument("--steps", type=int, default=32000, help="environment-slot budget per algorithm")
    p.add_argument("--eval-interval", type=int, default=2000, help="fixed-eval interval in environment slots")
    p.add_argument("--eval-slots", type=int, default=500, help="slots per deterministic fixed evaluation")
    p.add_argument("--output-root", default="results/rl_reward_comparison")
    p.add_argument("--tag", default=None, help="output subdirectory; defaults from --steps")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    requested = tuple(x.strip().upper() for x in args.algorithms.split(",") if x.strip())
    unknown = set(requested) - set(ALGORITHMS)
    if unknown:
        raise SystemExit(f"unknown algorithms: {sorted(unknown)}")
    if args.steps <= 0 or args.eval_interval <= 0 or args.eval_slots <= 0:
        raise SystemExit("steps/eval-interval/eval-slots must be positive")

    cfg = make_config(args.steps, args.eval_slots)
    tag = args.tag or output_tag(args.steps)
    out_dir = Path(args.output_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    histories: Dict[str, Dict] = {}
    summaries: Dict[str, Dict] = {}
    for alg in requested:
        history, summary = run_algorithm(alg, cfg, out_dir, args.steps,
                                         args.eval_interval, args.eval_slots, args.resume)
        histories[alg] = history; summaries[alg] = summary

    if set(requested) == set(ALGORITHMS):
        steps_ref = histories[ALGORITHMS[0]]["environment_steps"]
        for alg in ALGORITHMS[1:]:
            if histories[alg]["environment_steps"] != steps_ref:
                raise AssertionError("algorithms have mismatched evaluation points")
        write_comparison(out_dir, histories)
    manifest = make_manifest(cfg, args, out_dir, summaries)
    atomic_json(out_dir / "experiment_manifest.json", manifest)
    print(f"results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
