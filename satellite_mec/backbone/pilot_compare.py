"""One-seed mini-pilot for the four MARL backbones.

This runner is deliberately separate from the final 5-seed/32K experiment. It checks
that learning curves can be collected under a common protocol before spending the full
compute budget.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from core import Config, SatelliteMECEnv
from backbone.compare_algorithms import (
    SharedRewardMAPPO, MicroStepQMIXPolicy,
    seed_everything, system_reward, module_finite,
)
from backbone.fairness_fixes import FairIPPOPolicy, FairSharedRewardMADDPG

ALGO = os.environ.get("ALGORITHM", "MAPPO").upper()
SEED = int(os.environ.get("PILOT_SEED", "0"))
TRAIN_SLOTS = int(os.environ.get("PILOT_TRAIN_SLOTS", "1024"))
EVAL_INTERVAL = int(os.environ.get("PILOT_EVAL_INTERVAL", "256"))
EVAL_SLOTS = int(os.environ.get("PILOT_EVAL_SLOTS", "64"))
OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "backbone_compare" / "mini_pilot"


class PilotConfig(Config):
    """Controlled 25-satellite backbone-comparison configuration."""
    T_WARMUP = 0
    N_EVAL_RUNS = 1
    K_ROLLOUT = 64
    BETA_TASK = 0.0
    T_TOTAL = max(TRAIN_SLOTS + 128, EVAL_SLOTS + 128, 2048)


def paired_train_seeds(seed: int):
    base = 10_000 + 100 * seed
    return {"task": base, "task_param": base + 1, "dod_init": base + 2}


def fixed_eval_seeds():
    return {"task": 91_001, "task_param": 91_002, "dod_init": 91_003}


def common_reward_from_info(info: dict, n_agents: int) -> float:
    """Reconstruct the same full reward in train and eval phases from the ledger."""
    ledger = info.get("reward_ledger", {})
    keys = ("action_cost", "done", "timeout", "reject", "hl", "queue", "dod")
    if ledger:
        return float(sum(float(ledger.get(k, 0.0)) for k in keys) / max(n_agents, 1))
    return 0.0


def build_policy(name: str, cfg, env):
    if name == "MAPPO":
        p = SharedRewardMAPPO(cfg, name="MAPPO")
    elif name == "IPPO":
        p = FairIPPOPolicy(cfg, name="IPPO")
    elif name == "MADDPG":
        p = FairSharedRewardMADDPG(
            cfg, env, name="MADDPG", seed=SEED,
            actor_lr=1e-3, critic_lr=1e-3, gamma=cfg.GAMMA, tau=0.01,
            batch_size=128, start_steps=256, updates_per_slot=1,
            replay_size=100_000, reward_scale=1.0,
            tau_gs_start=1.0, tau_gs_end=0.5,
            anneal_slots=max(TRAIN_SLOTS, 1), expl_noise=0.2,
            zhong_reward=False,
        )
    elif name == "QMIX":
        p = MicroStepQMIXPolicy(
            cfg, seed=SEED, lr=5e-4, batch_size=32, replay_size=5000,
            start_transitions=128, target_interval=200,
            eps_start=1.0, eps_end=0.05, eps_anneal_slots=max(TRAIN_SLOTS, 1),
        )
    else:
        raise ValueError(f"unsupported algorithm: {name}")
    p.set_train_mode()
    return p


def update_count(policy, name: str) -> int:
    if name in ("MAPPO", "IPPO"):
        return int(policy.trainer.update_count)
    if name == "MADDPG":
        return int(policy.backbone_update_count)
    return int(policy.update_count)


def params_finite(policy, name: str) -> bool:
    if name in ("MAPPO", "IPPO"):
        return module_finite(policy.actor) and module_finite(policy.critic)
    if name == "MADDPG":
        return module_finite(policy.actor) and module_finite(policy.critic)
    return policy.parameters_finite()


def evaluate(policy, cfg) -> dict:
    eval_env = SatelliteMECEnv(cfg)
    eval_env.reset("eval", fixed_eval_seeds())
    policy.set_eval_mode()
    returns = []
    slot_rewards = []
    info = {}
    try:
        for _ in range(EVAL_SLOTS):
            _rewards, _done, info = policy.run_step(eval_env)
            slot_rewards.append(common_reward_from_info(info, cfg.N_SATS))
        k = cfg.K_ROLLOUT
        for start in range(0, len(slot_rewards), k):
            chunk = slot_rewards[start:start + k]
            if chunk:
                returns.append(float(np.sum(chunk)))
    finally:
        policy.set_train_mode()
    return {
        "mean_eval_episode_return": float(np.mean(returns)) if returns else 0.0,
        "mean_eval_slot_Rsys": float(np.mean(slot_rewards)) if slot_rewards else 0.0,
        "eval_completion_rate": float(eval_env.get_eval_completion_rate()),
        "eval_avg_dod": float(info.get("avg_dod", 0.0)),
        "eval_avg_health_loss": float(info.get("avg_health_loss", 0.0)),
    }


def run() -> dict:
    if ALGO not in ("MAPPO", "IPPO", "MADDPG", "QMIX"):
        raise ValueError(ALGO)
    seed_everything(SEED)
    cfg = PilotConfig()
    cfg.T_TRAIN = TRAIN_SLOTS
    cfg.T_EVAL = EVAL_SLOTS
    env = SatelliteMECEnv(cfg)
    env.reset("train", paired_train_seeds(SEED))
    policy = build_policy(ALGO, cfg, env)

    checkpoints = []
    train_slot_rewards = []
    t0 = time.time()
    for slot in range(1, TRAIN_SLOTS + 1):
        rewards, _done, _info = policy.run_step(env)
        r = system_reward(rewards)
        if not math.isfinite(r):
            raise RuntimeError(f"{ALGO}: non-finite train reward at slot {slot}")
        train_slot_rewards.append(r)
        if slot == 1 or slot % EVAL_INTERVAL == 0 or slot == TRAIN_SLOTS:
            ev = evaluate(policy, cfg)
            row = {
                "train_slots": slot,
                "updates": update_count(policy, ALGO),
                "mean_recent_train_Rsys": float(np.mean(train_slot_rewards[-EVAL_INTERVAL:])),
                **ev,
            }
            checkpoints.append(row)
            print(json.dumps(row), flush=True)

    if ALGO in ("QMIX", "MADDPG"):
        policy.finalize_training()
    if not params_finite(policy, ALGO):
        raise RuntimeError(f"{ALGO}: non-finite parameters")
    if ALGO == "QMIX" and policy.invalid_actions:
        raise RuntimeError(f"QMIX invalid actions: {policy.invalid_actions}")

    elapsed = time.time() - t0
    result = {
        "algorithm": ALGO,
        "seed": SEED,
        "status": "PASS",
        "protocol": {
            "n_sats": cfg.N_SATS,
            "train_slots": TRAIN_SLOTS,
            "rollout_slots": cfg.K_ROLLOUT,
            "eval_interval_slots": EVAL_INTERVAL,
            "eval_slots": EVAL_SLOTS,
            "beta_task": cfg.BETA_TASK,
            "gamma": cfg.GAMMA,
            "shared_reward": "R_sys = mean_n r_n",
            "deterministic_fixed_seed_evaluation": True,
            "evaluation_reward_reconstructed_from_ledger": True,
            "paired_training_environment_seeds": paired_train_seeds(SEED),
            "fixed_evaluation_environment_seeds": fixed_eval_seeds(),
        },
        "checkpoints": checkpoints,
        "final_updates": update_count(policy, ALGO),
        "wall_time_sec": elapsed,
        "all_params_finite": True,
    }
    if ALGO == "IPPO":
        result["ippo"] = {
            "critic_input": "task-free actor-observable context: own + 1-hop neighbours (47 dims)",
            "centralized_global_summary": False,
        }
    if ALGO == "QMIX":
        result["qmix"] = {
            "adapter": "decision-round micro-step with internal NOOP",
            "within_slot_discount": 1.0,
            "inter_slot_discount": cfg.GAMMA,
            "replay_size": len(policy.replay),
            "invalid_actions": policy.invalid_actions,
            "last_loss": policy.last_loss,
        }
    if ALGO == "MADDPG":
        result["maddpg"] = {
            "discrete_actions": "Gumbel-Softmax",
            "zhong_reward": False,
            "common_Rsys": True,
            "slot_reward_conservation": "R_sys divided across task replay transitions",
            "rollout_boundary_terminal_convention": False,
            "final_unmatched_transition": "discarded (continuing environment)",
        }
    return result


def main():
    result = run()
    out_dir = OUT_ROOT / ALGO / f"seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pilot.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"WROTE {path}", flush=True)

if __name__ == "__main__":
    main()
