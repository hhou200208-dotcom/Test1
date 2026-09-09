"""Formal one-seed 32K MARL experiment with unified train/eval metrics.

Algorithms
----------
BLA-MAPPO, MAPPO, IPPO, MADDPG, QMIX.

Every algorithm is trained in the same SatelliteMECEnv, with the same environment seed,
shared system reward definition and physical metric logger.  The script records:

* training 64-slot return + CR + delay + energy + cumulative HL;
* deterministic evaluation return + CR + delay + energy + cumulative HL;
* DoD and supporting counts so every aggregate can be audited later.

This is one paired training seed only.  It is an engineering/formalization run before the
final 5-seed statistical comparison.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from core import Config, SatelliteMECEnv
from backbone.compare_algorithms import (
    SharedRewardMAPPO,
    MicroStepQMIXPolicy,
    module_finite,
    seed_everything,
    system_reward,
)
from backbone.fairness_fixes import FairIPPOPolicy, FairSharedRewardMADDPG
from backbone.pilot_compare import paired_train_seeds, fixed_eval_seeds, common_reward_from_info

ALGO_INPUT = os.environ.get("ALGORITHM", "MAPPO").upper().replace("-", "_")
SEED = int(os.environ.get("FORMAL_SEED", "0"))
TRAIN_SLOTS = int(os.environ.get("FORMAL_TRAIN_SLOTS", "32000"))
ROLLOUT_SLOTS = int(os.environ.get("FORMAL_ROLLOUT_SLOTS", "64"))
EVAL_INTERVAL = int(os.environ.get("FORMAL_EVAL_INTERVAL", "640"))
EVAL_SLOTS = int(os.environ.get("FORMAL_EVAL_SLOTS", "64"))
OUT_ROOT = (Path(__file__).resolve().parents[1] / "results" / "backbone_compare" /
            "formal_32k_seed0")

DISPLAY = {
    "BLA_MAPPO": "BLA-MAPPO",
    "MAPPO": "MAPPO",
    "IPPO": "IPPO",
    "MADDPG": "MADDPG",
    "QMIX": "QMIX",
}

# Frozen from the equal-budget pilot for the four backbones.  BLA-MAPPO deliberately
# shares MAPPO's optimizer settings; the only comparison-specific change is task credit.
SELECTED = {
    "BLA_MAPPO": {"actor_lr": 3e-4, "critic_lr": 1e-3, "beta_task": 0.5},
    "MAPPO": {"actor_lr": 3e-4, "critic_lr": 1e-3, "beta_task": 0.0},
    "IPPO": {"actor_lr": 3e-4, "critic_lr": 1e-3, "beta_task": 0.0},
    "MADDPG": {"actor_lr": 1e-4, "critic_lr": 3e-4, "expl_noise": 0.15},
    "QMIX": {"lr": 5e-4},
}


class FormalConfig(Config):
    T_WARMUP = 0
    N_EVAL_RUNS = 1
    K_ROLLOUT = ROLLOUT_SLOTS
    T_TOTAL = max(TRAIN_SLOTS + 512, EVAL_SLOTS + 512, 33000)


def build_policy(name: str, cfg, env):
    hp = SELECTED[name]
    if name in ("BLA_MAPPO", "MAPPO"):
        cfg.LR_ACTOR = hp["actor_lr"]
        cfg.LR_CRITIC = hp["critic_lr"]
        cfg.BETA_TASK = hp["beta_task"]
        p = SharedRewardMAPPO(cfg, name=DISPLAY[name])
    elif name == "IPPO":
        cfg.LR_ACTOR = hp["actor_lr"]
        cfg.LR_CRITIC = hp["critic_lr"]
        cfg.BETA_TASK = 0.0
        p = FairIPPOPolicy(cfg, name="IPPO")
    elif name == "MADDPG":
        cfg.BETA_TASK = 0.0
        p = FairSharedRewardMADDPG(
            cfg, env, name="MADDPG", seed=SEED,
            actor_lr=hp["actor_lr"], critic_lr=hp["critic_lr"],
            gamma=cfg.GAMMA, tau=0.01, batch_size=128, start_steps=256,
            updates_per_slot=1, replay_size=250_000, reward_scale=1.0,
            tau_gs_start=1.0, tau_gs_end=0.5, anneal_slots=max(TRAIN_SLOTS, 1),
            expl_noise=hp["expl_noise"], zhong_reward=False,
        )
    elif name == "QMIX":
        cfg.BETA_TASK = 0.0
        p = MicroStepQMIXPolicy(
            cfg, seed=SEED, lr=hp["lr"], batch_size=32, replay_size=15000,
            start_transitions=128, target_interval=200,
            eps_start=1.0, eps_end=0.05, eps_anneal_slots=max(TRAIN_SLOTS, 1),
        )
    else:
        raise ValueError(name)
    p.set_train_mode()
    return p


def update_count(policy, name: str) -> int:
    if name in ("BLA_MAPPO", "MAPPO", "IPPO"):
        return int(policy.trainer.update_count)
    if name == "MADDPG":
        return int(policy.backbone_update_count)
    return int(policy.update_count)


def params_finite(policy, name: str) -> bool:
    if name in ("BLA_MAPPO", "MAPPO", "IPPO"):
        return module_finite(policy.actor) and module_finite(policy.critic)
    if name == "MADDPG":
        return module_finite(policy.actor) and module_finite(policy.critic)
    return policy.parameters_finite() and policy.invalid_actions == 0


def _delay_samples(infos: Iterable[Dict]) -> List[float]:
    vals: List[float] = []
    for info in infos:
        vals.extend(float(x) for x in info.get("slot_e2e_delays", []))
    return vals


def aggregate_window(rewards: Iterable[float], infos: List[Dict], n_sats: int,
                     prefix: str) -> Dict:
    """Aggregate one physical window without changing the simulator definition.

    Delay is the E2E delay of completed tasks (same physical samples used by the existing
    MetricsRecorder).  Energy is compute+transmit system energy; housekeeping energy is
    intentionally excluded because `slot_system_energy` excludes it for policy comparison.
    `cum_hl_mean_satellite` matches the repository's existing cumulative-HL convention:
    sum over slots of per-slot average satellite health loss.  System-total HL is retained
    separately as `cum_hl_system` (= mean-satellite convention * N).
    """
    rewards = list(rewards)
    total_arrived = int(sum(int(i.get("arrived", 0)) for i in infos))
    total_done = int(sum(int(i.get("done_tasks", 0)) for i in infos))
    total_timeout = int(sum(int(i.get("slot_timeout", 0)) for i in infos))
    total_rejected = int(sum(int(i.get("rejected", 0)) for i in infos))
    delays = _delay_samples(infos)
    total_energy = float(sum(float(i.get("slot_system_energy", 0.0)) for i in infos))
    total_energy_comp = float(sum(float(i.get("slot_system_energy_comp", 0.0)) for i in infos))
    total_energy_trans = float(sum(float(i.get("slot_system_energy_trans", 0.0)) for i in infos))
    cum_hl_mean_sat = float(sum(float(i.get("avg_health_loss", 0.0)) for i in infos))
    avg_dod = float(np.mean([float(i.get("avg_dod", 0.0)) for i in infos])) if infos else 0.0
    n_slots = len(infos)
    return {
        f"{prefix}_return": float(np.sum(rewards)),
        f"{prefix}_mean_slot_Rsys": float(np.mean(rewards)) if rewards else 0.0,
        f"{prefix}_cr": float(total_done / max(total_arrived, 1)),
        f"{prefix}_arrived": total_arrived,
        f"{prefix}_done": total_done,
        f"{prefix}_timeout": total_timeout,
        f"{prefix}_rejected": total_rejected,
        f"{prefix}_avg_delay_s": float(np.mean(delays)) if delays else 0.0,
        f"{prefix}_p95_delay_s": float(np.percentile(delays, 95)) if delays else 0.0,
        f"{prefix}_delay_samples": len(delays),
        f"{prefix}_total_energy_j": total_energy,
        f"{prefix}_energy_comp_j": total_energy_comp,
        f"{prefix}_energy_trans_j": total_energy_trans,
        f"{prefix}_avg_energy_j_per_slot": total_energy / max(n_slots, 1),
        f"{prefix}_cum_hl_mean_satellite": cum_hl_mean_sat,
        f"{prefix}_cum_hl_system": cum_hl_mean_sat * n_sats,
        f"{prefix}_avg_hl_per_sat_slot": cum_hl_mean_sat / max(n_slots, 1),
        f"{prefix}_avg_dod": avg_dod,
        f"{prefix}_n_slots": n_slots,
    }


def evaluate(policy, cfg, checkpoint_slot: int) -> Dict:
    env = SatelliteMECEnv(cfg)
    env.reset("eval", fixed_eval_seeds())
    policy.set_eval_mode()
    rewards: List[float] = []
    infos: List[Dict] = []
    try:
        for _ in range(EVAL_SLOTS):
            _r, _done, info = policy.run_step(env)
            rs = common_reward_from_info(info, cfg.N_SATS)
            if not math.isfinite(rs):
                raise RuntimeError("non-finite evaluation reward")
            rewards.append(rs)
            infos.append(info)
    finally:
        policy.set_train_mode()
    out = {"train_slots": int(checkpoint_slot)}
    out.update(aggregate_window(rewards, infos, cfg.N_SATS, "eval"))
    # Explicitly duplicate the environment's run-level CR for an audit check.
    out["eval_cr_env_counter"] = float(env.get_eval_completion_rate())
    if abs(out["eval_cr"] - out["eval_cr_env_counter"]) > 1e-12:
        raise RuntimeError(
            f"evaluation CR mismatch: aggregate={out['eval_cr']} env={out['eval_cr_env_counter']}")
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    if ALGO_INPUT not in SELECTED:
        raise ValueError(f"unsupported ALGORITHM={ALGO_INPUT}")
    name = ALGO_INPUT
    display = DISPLAY[name]
    seed_everything(SEED)
    cfg = FormalConfig()
    cfg.T_TRAIN = TRAIN_SLOTS
    cfg.T_EVAL = EVAL_SLOTS
    env = SatelliteMECEnv(cfg)
    env.reset("train", paired_train_seeds(SEED))
    policy = build_policy(name, cfg, env)

    train_rollouts: List[Dict] = []
    eval_checkpoints: List[Dict] = []
    slot_rewards: List[float] = []
    slot_infos: List[Dict] = []
    t0 = time.time()

    # Initial deterministic evaluation gives a common x=0 baseline for the curve.
    initial = evaluate(policy, cfg, 0)
    initial["updates"] = update_count(policy, name)
    eval_checkpoints.append(initial)
    print(json.dumps({"algorithm": display, "checkpoint": initial}), flush=True)

    for slot in range(1, TRAIN_SLOTS + 1):
        rewards, _done, info = policy.run_step(env)
        rs = system_reward(rewards)
        if not math.isfinite(rs):
            raise RuntimeError(f"{display}: non-finite training reward at slot {slot}")
        slot_rewards.append(rs)
        slot_infos.append(info)

        if slot % ROLLOUT_SLOTS == 0 or slot == TRAIN_SLOTS:
            row = {
                "train_slots": slot,
                "rollout_index": len(train_rollouts),
                "updates": update_count(policy, name),
            }
            row.update(aggregate_window(slot_rewards, slot_infos, cfg.N_SATS, "train"))
            train_rollouts.append(row)
            slot_rewards.clear()
            slot_infos.clear()

        if slot % EVAL_INTERVAL == 0 or slot == TRAIN_SLOTS:
            ev = evaluate(policy, cfg, slot)
            ev["updates"] = update_count(policy, name)
            eval_checkpoints.append(ev)
            print(json.dumps({"algorithm": display, "checkpoint": ev}), flush=True)

    if name in ("MADDPG", "QMIX"):
        policy.finalize_training()
    if not params_finite(policy, name):
        raise RuntimeError(f"{display}: non-finite/invalid model after training")

    elapsed = time.time() - t0
    final_eval = eval_checkpoints[-1]
    result = {
        "algorithm": display,
        "algorithm_key": name,
        "seed": SEED,
        "status": "PASS",
        "selected_hyperparameters": SELECTED[name],
        "protocol": {
            "n_sats": cfg.N_SATS,
            "train_slots": TRAIN_SLOTS,
            "rollout_slots": ROLLOUT_SLOTS,
            "eval_interval_slots": EVAL_INTERVAL,
            "eval_slots": EVAL_SLOTS,
            "gamma": cfg.GAMMA,
            "ppo_entropy_beta": cfg.BETA,
            "beta_task": float(getattr(cfg, "BETA_TASK", 0.0)),
            "shared_reward": "R_sys(t) = mean_n r_n(t)",
            "train_return": "sum of R_sys over each 64-slot training rollout",
            "eval_return": "sum of reconstructed full R_sys over fixed-seed deterministic evaluation",
            "cr": "completed / arrived within the corresponding metric window",
            "delay": "mean E2E delay of completed tasks in seconds",
            "energy": "system compute + transmit energy; housekeeping excluded by environment metric",
            "cum_hl_mean_satellite": "sum over slots of avg_health_loss across satellites",
            "cum_hl_system": "N_SATS * cum_hl_mean_satellite",
            "paired_train_seeds": paired_train_seeds(SEED),
            "fixed_eval_seeds": fixed_eval_seeds(),
        },
        "train_rollouts": train_rollouts,
        "eval_checkpoints": eval_checkpoints,
        "final_eval": final_eval,
        "final_updates": update_count(policy, name),
        "wall_time_sec": elapsed,
        "all_params_finite": True,
    }
    if name == "QMIX":
        result["qmix"] = {
            "microstep": "kth task decision round + internal NOOP",
            "replay_size": len(policy.replay),
            "last_loss": policy.last_loss,
            "invalid_actions": policy.invalid_actions,
        }

    out_dir = OUT_ROOT / name / f"seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out_dir / "train_rollouts.csv", train_rollouts)
    write_csv(out_dir / "eval_checkpoints.csv", eval_checkpoints)
    print(f"WROTE {out_dir}", flush=True)
    print(json.dumps({"algorithm": display, "final_eval": final_eval,
                      "wall_time_sec": elapsed}), flush=True)


if __name__ == "__main__":
    main()
