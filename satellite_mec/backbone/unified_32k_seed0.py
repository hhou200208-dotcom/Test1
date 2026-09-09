"""Unified formal benchmark: 25 satellites, 32K training, one paired seed.

This runner freezes the configuration agreed for the paper-facing MARL comparison.
All five learning methods see the same SatelliteMECEnv, task load, physical model,
external reward, training budget, train seeds and evaluation seeds.

Online convergence evaluation is short and fixed-seed.  The final performance table
is produced from the frozen 32K parameter state with a separate 5400-slot evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from core import Config, SatelliteMECEnv
from backbone.compare_algorithms import (
    SharedRewardMAPPO,
    MicroStepQMIXPolicy,
    module_finite,
    seed_everything,
    system_reward,
)
from backbone.fairness_fixes import FairIPPOPolicy, FairSharedRewardMADDPG
from backbone.pilot_compare import common_reward_from_info

ALGO_INPUT = os.environ.get("ALGORITHM", "MAPPO").upper().replace("-", "_")
SEED = int(os.environ.get("UNIFIED_SEED", "0"))
TRAIN_SLOTS = int(os.environ.get("UNIFIED_TRAIN_SLOTS", "32000"))
ROLLOUT_SLOTS = int(os.environ.get("UNIFIED_ROLLOUT_SLOTS", "64"))
ONLINE_EVAL_INTERVAL = int(os.environ.get("UNIFIED_EVAL_INTERVAL", "640"))
ONLINE_EVAL_SLOTS = int(os.environ.get("UNIFIED_ONLINE_EVAL_SLOTS", "64"))
FINAL_EVAL_SLOTS = int(os.environ.get("UNIFIED_FINAL_EVAL_SLOTS", "5400"))
RUN_TAG = os.environ.get("UNIFIED_RUN_TAG", "unified_32k_seed0")
OUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "backbone_compare" / RUN_TAG

DISPLAY = {
    "BLA_MAPPO": "BLA-MAPPO",
    "MAPPO": "MAPPO",
    "IPPO": "IPPO",
    "MADDPG": "MADDPG",
    "QMIX": "QMIX",
}

# PPO family uses the finalized paper recipe.  Off-policy backbones retain the
# equal-budget selections from the earlier tuning stage; these are algorithm-specific
# optimizer choices, not environment/reward differences.
SELECTED = {
    "BLA_MAPPO": {"actor_lr": 1e-4, "critic_lr": 1e-3, "beta_task": 0.5},
    "MAPPO": {"actor_lr": 1e-4, "critic_lr": 1e-3, "beta_task": 0.0},
    "IPPO": {"actor_lr": 1e-4, "critic_lr": 1e-3, "beta_task": 0.0},
    "MADDPG": {"actor_lr": 1e-4, "critic_lr": 3e-4, "expl_noise": 0.15},
    "QMIX": {"lr": 5e-4},
}


class UnifiedPaperConfig(Config):
    """Single frozen environment/reward configuration for all five algorithms."""

    # 25-satellite subscale constellation used by the raw simulator.
    N_PLANES = 5
    N_SATS_PER_PLANE = 5
    N_SATS = 25

    # Final workload used by the main experiments: 1/5 hotspots at lambda_H=4.
    LAMBDA_HIGH = 4.0
    LAMBDA_LOW = 0.1
    LAMBDA_HIGH_RATIO = 1 / 5
    LAMBDA = LAMBDA_HIGH * LAMBDA_HIGH_RATIO + LAMBDA_LOW * (1 - LAMBDA_HIGH_RATIO)

    # Task/compute model retained from the final DVFS-era implementation.
    H_MIN = 10.0
    H_MAX = 30.0
    CPU_FREQ = 2e9
    MAX_DISPATCH = 6

    # Physical parameters frozen to the main-figure setup agreed for this rerun.
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0
    DOD_MAX = 0.8
    P_SOLAR_MAX = 30.0
    P_HOUSEKEEPING = 5.0

    # Lyapunov and finalized external reward.
    V = 50.0
    ETA = 0.5
    W_DONE = 10.0
    W_TIMEOUT = 5.0
    W_REJECT = 5.0
    W_HL = 2.0
    W_QUEUE = 0.05

    # Final PPO recipe shared by BLA-MAPPO/MAPPO/IPPO except task credit/critic scope.
    GAMMA = 0.99
    LAMBDA_GAE = 0.95
    EPSILON = 0.2
    BETA = 0.02
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    MINIBATCH = 64
    EPOCH = 2
    K_ROLLOUT = ROLLOUT_SLOTS

    T_WARMUP = 0
    N_EVAL_RUNS = 1

    def __init__(self):
        super().__init__()
        # The main-figure physical setup used this DVFS trade-off coefficient.
        # Config.__init__ auto-calibrates a value, so freeze the agreed value after
        # all other derived quantities have been computed and before env creation.
        self.V_DVFS = 2.0e17
        self.T_TRAIN = TRAIN_SLOTS
        self.T_EVAL = FINAL_EVAL_SLOTS
        self.T_TOTAL = max(TRAIN_SLOTS + 1024, FINAL_EVAL_SLOTS + 1024)


def train_seeds(seed_idx: int) -> Dict[str, int]:
    # Seed-0 matches the repository's canonical 42/43/44 task stream convention.
    base = 42 + 1000 * seed_idx
    return {"task": base, "task_param": base + 1, "dod_init": base + 2}


def optimizer_seed(seed_idx: int) -> int:
    return 46 + 1000 * seed_idx


def online_eval_seeds(cfg) -> Dict[str, int]:
    # Same fixed quick-eval convention used by the main MAPPO training pipeline.
    return cfg.get_quick_eval_seeds()


def final_eval_seeds(cfg) -> Dict[str, int]:
    # Same run-0 evaluation seed convention used by the paper evaluation scripts.
    return cfg.get_eval_seeds(0)


def build_policy(name: str, cfg, env):
    hp = SELECTED[name]
    alg_seed = optimizer_seed(SEED)
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
            cfg, env, name="MADDPG", seed=alg_seed,
            actor_lr=hp["actor_lr"], critic_lr=hp["critic_lr"],
            gamma=cfg.GAMMA, tau=0.01, batch_size=128, start_steps=256,
            updates_per_slot=1, replay_size=250_000, reward_scale=1.0,
            tau_gs_start=1.0, tau_gs_end=0.5, anneal_slots=max(TRAIN_SLOTS, 1),
            expl_noise=hp["expl_noise"], zhong_reward=False,
        )
    elif name == "QMIX":
        cfg.BETA_TASK = 0.0
        p = MicroStepQMIXPolicy(
            cfg, seed=alg_seed, lr=hp["lr"], batch_size=32, replay_size=15000,
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
    """Aggregate using the paper metric definitions fixed in this conversation.

    satisfaction = deadline-completed tasks / all task arrivals.
    system_delay_overhead = sum of completed-task E2E delays / physical slots [s/slot].
    system_energy_overhead = dynamic compute+ISL energy / slots [kJ/slot].
    cumulative_lifetime_loss = sum over both satellite and time dimensions.
    """
    rewards = list(rewards)
    total_arrived = int(sum(int(i.get("arrived", 0)) for i in infos))
    total_done = int(sum(int(i.get("done_tasks", 0)) for i in infos))
    total_timeout = int(sum(int(i.get("slot_timeout", 0)) for i in infos))
    total_rejected = int(sum(int(i.get("rejected", 0)) for i in infos))
    delays = _delay_samples(infos)
    delay_sum = float(np.sum(delays)) if delays else 0.0
    total_energy_j = float(sum(float(i.get("slot_system_energy", 0.0)) for i in infos))
    total_energy_comp_j = float(sum(float(i.get("slot_system_energy_comp", 0.0)) for i in infos))
    total_energy_trans_j = float(sum(float(i.get("slot_system_energy_trans", 0.0)) for i in infos))
    cum_hl_mean_sat = float(sum(float(i.get("avg_health_loss", 0.0)) for i in infos))
    cum_hl_system = cum_hl_mean_sat * n_sats
    n_slots = len(infos)
    avg_dod = float(np.mean([float(i.get("avg_dod", 0.0)) for i in infos])) if infos else 0.0
    satisfaction = float(total_done / max(total_arrived, 1))
    return {
        f"{prefix}_return": float(np.sum(rewards)),
        f"{prefix}_mean_slot_Rsys": float(np.mean(rewards)) if rewards else 0.0,
        f"{prefix}_satisfaction": satisfaction,
        f"{prefix}_cr": satisfaction,
        f"{prefix}_arrived": total_arrived,
        f"{prefix}_done": total_done,
        f"{prefix}_timeout": total_timeout,
        f"{prefix}_rejected": total_rejected,
        f"{prefix}_system_delay_overhead_s_per_slot": delay_sum / max(n_slots, 1),
        f"{prefix}_delay_sum_s": delay_sum,
        f"{prefix}_avg_completed_e2e_delay_s": float(np.mean(delays)) if delays else 0.0,
        f"{prefix}_p95_completed_e2e_delay_s": float(np.percentile(delays, 95)) if delays else 0.0,
        f"{prefix}_delay_samples": len(delays),
        f"{prefix}_system_energy_overhead_kj_per_slot": total_energy_j / max(n_slots, 1) / 1000.0,
        f"{prefix}_total_dynamic_energy_j": total_energy_j,
        f"{prefix}_energy_comp_j": total_energy_comp_j,
        f"{prefix}_energy_trans_j": total_energy_trans_j,
        f"{prefix}_cumulative_lifetime_loss": cum_hl_system,
        f"{prefix}_cum_hl_mean_satellite": cum_hl_mean_sat,
        f"{prefix}_avg_hl_per_sat_slot": cum_hl_mean_sat / max(n_slots, 1),
        f"{prefix}_avg_dod": avg_dod,
        f"{prefix}_n_slots": n_slots,
    }


def evaluate(policy, cfg, checkpoint_slot: int, n_slots: int, seeds: Dict[str, int],
             prefix: str) -> Dict:
    env = SatelliteMECEnv(cfg)
    env.reset("eval", seeds)
    policy.set_eval_mode()
    rewards: List[float] = []
    infos: List[Dict] = []
    try:
        for _ in range(n_slots):
            _r, _done, info = policy.run_step(env)
            rs = common_reward_from_info(info, cfg.N_SATS)
            if not math.isfinite(rs):
                raise RuntimeError("non-finite evaluation reward")
            rewards.append(rs)
            infos.append(info)
    finally:
        policy.set_train_mode()
    out = {"train_slots": int(checkpoint_slot)}
    out.update(aggregate_window(rewards, infos, cfg.N_SATS, prefix))
    env_cr = float(env.get_eval_completion_rate())
    out[f"{prefix}_cr_env_counter"] = env_cr
    # Both definitions are done/arrived for a fresh evaluation trajectory.
    if abs(out[f"{prefix}_cr"] - env_cr) > 1e-12:
        raise RuntimeError(
            f"CR mismatch: aggregate={out[f'{prefix}_cr']} env={env_cr}")
    return out


def save_checkpoint(policy, name: str, path: Path, cfg) -> str:
    """Save a self-describing state-dict checkpoint without relying on wrapper APIs."""
    attrs = [
        "actor", "critic", "actor_target", "critic_target",
        "agent", "target_agent", "mixer", "target_mixer",
        "actor_opt", "critic_opt", "optimizer", "optim",
    ]
    state = {
        "algorithm": DISPLAY[name],
        "algorithm_key": name,
        "seed": SEED,
        "train_slots": TRAIN_SLOTS,
        "updates": update_count(policy, name),
        "config": config_fingerprint(cfg),
        "modules": {},
    }
    for attr in attrs:
        obj = getattr(policy, attr, None)
        if obj is not None and hasattr(obj, "state_dict"):
            try:
                state["modules"][attr] = obj.state_dict()
            except Exception:
                pass
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_fingerprint(cfg) -> Dict:
    return {
        "n_sats": cfg.N_SATS,
        "n_planes": cfg.N_PLANES,
        "n_sats_per_plane": cfg.N_SATS_PER_PLANE,
        "tau_s": cfg.TAU,
        "orbit_period_slots": cfg.ORBIT_PERIOD,
        "lambda_high": cfg.LAMBDA_HIGH,
        "lambda_low": cfg.LAMBDA_LOW,
        "lambda_high_ratio": cfg.LAMBDA_HIGH_RATIO,
        "lambda_mean": cfg.LAMBDA,
        "task_size_bits": [cfg.S_MIN, cfg.S_MAX],
        "cycles_per_bit": [cfg.H_MIN, cfg.H_MAX],
        "deadline_s": [cfg.D_MAX_MIN, cfg.D_MAX_MAX],
        "max_hops": cfg.K_MAX,
        "cpu_freq_max_hz": cfg.CPU_FREQ,
        "kappa": cfg.KAPPA,
        "v_dvfs": cfg.V_DVFS,
        "isl_rate_bps": [cfg.B_MIN, cfg.B_MAX],
        "tx_power_w": cfg.P_T,
        "battery_capacity_j": cfg.E_CAP,
        "solar_max_w": cfg.P_SOLAR_MAX,
        "housekeeping_w": cfg.P_HOUSEKEEPING,
        "dod_range": [cfg.DOD_MIN, cfg.DOD_MAX],
        "dod_init_range": [cfg.DOD_INIT_LOW, cfg.DOD_INIT_HIGH],
        "a_coef": cfg.A_COEF,
        "lyapunov_v": cfg.V,
        "lyapunov_eta": cfg.ETA,
        "reward": {
            "done": cfg.W_DONE,
            "timeout": cfg.W_TIMEOUT,
            "reject": cfg.W_REJECT,
            "hl": cfg.W_HL,
            "queue": cfg.W_QUEUE,
            "hl_norm": cfg.HL_NORM,
        },
        "ppo": {
            "gamma": cfg.GAMMA,
            "gae_lambda": cfg.LAMBDA_GAE,
            "clip": cfg.EPSILON,
            "entropy_beta": cfg.BETA,
            "minibatch": cfg.MINIBATCH,
            "epochs": cfg.EPOCH,
            "rollout_slots": cfg.K_ROLLOUT,
        },
    }


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

    cfg = UnifiedPaperConfig()
    seed_everything(optimizer_seed(SEED))
    env = SatelliteMECEnv(cfg)
    tr_seeds = train_seeds(SEED)
    env.reset("train", tr_seeds)
    policy = build_policy(name, cfg, env)

    out_dir = OUT_ROOT / name / f"seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rollouts: List[Dict] = []
    online_eval: List[Dict] = []
    slot_rewards: List[float] = []
    slot_infos: List[Dict] = []
    t0 = time.time()

    # Common deterministic x=0 point.
    initial = evaluate(
        policy, cfg, 0, ONLINE_EVAL_SLOTS, online_eval_seeds(cfg), "eval"
    )
    initial["updates"] = update_count(policy, name)
    online_eval.append(initial)
    print(json.dumps({"algorithm": display, "online_eval": initial}), flush=True)

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

        if slot % ONLINE_EVAL_INTERVAL == 0 or slot == TRAIN_SLOTS:
            ev = evaluate(
                policy, cfg, slot, ONLINE_EVAL_SLOTS, online_eval_seeds(cfg), "eval"
            )
            ev["updates"] = update_count(policy, name)
            online_eval.append(ev)
            print(json.dumps({"algorithm": display, "online_eval": ev}), flush=True)

    if name in ("MADDPG", "QMIX"):
        policy.finalize_training()
    if not params_finite(policy, name):
        raise RuntimeError(f"{display}: non-finite/invalid model after training")

    # Freeze and save the exact 32K state BEFORE the long final evaluation.
    ckpt_path = out_dir / "checkpoint_32k.pt"
    checkpoint_sha256 = save_checkpoint(policy, name, ckpt_path, cfg)

    final_eval = evaluate(
        policy, cfg, TRAIN_SLOTS, FINAL_EVAL_SLOTS, final_eval_seeds(cfg), "final"
    )
    final_eval["updates"] = update_count(policy, name)
    elapsed = time.time() - t0

    result = {
        "algorithm": display,
        "algorithm_key": name,
        "seed": SEED,
        "status": "PASS",
        "selected_hyperparameters": SELECTED[name],
        "config": config_fingerprint(cfg),
        "protocol": {
            "train_slots": TRAIN_SLOTS,
            "rollout_slots": ROLLOUT_SLOTS,
            "online_eval_interval_slots": ONLINE_EVAL_INTERVAL,
            "online_eval_slots": ONLINE_EVAL_SLOTS,
            "final_eval_slots": FINAL_EVAL_SLOTS,
            "shared_external_reward": "R_sys(t)=mean_n r_n(t), identical environment reward for all algorithms",
            "training_metric_role": "diagnostic/exploratory policy",
            "online_eval_role": "fixed-seed deterministic convergence curve",
            "final_eval_role": "frozen 32K state, independent 5400-slot deterministic performance evaluation",
            "satisfaction": "deadline-completed / arrived",
            "system_delay_overhead": "sum(completed-task E2E delay) / slots [s/slot]",
            "system_energy_overhead": "sum(compute+ISL dynamic energy) / slots / 1000 [kJ/slot]",
            "cumulative_lifetime_loss": "sum over satellites and slots of task-induced health loss",
            "train_seeds": tr_seeds,
            "online_eval_seeds": online_eval_seeds(cfg),
            "final_eval_seeds": final_eval_seeds(cfg),
        },
        "train_rollouts": train_rollouts,
        "online_eval_checkpoints": online_eval,
        "final_5400_eval": final_eval,
        "final_updates": update_count(policy, name),
        "checkpoint_file": ckpt_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "wall_time_sec": elapsed,
        "all_params_finite": True,
    }
    if name == "QMIX":
        result["qmix"] = {
            "adaptation": "decision-round microstep with internal NOOP",
            "replay_size": len(policy.replay),
            "last_loss": policy.last_loss,
            "invalid_actions": policy.invalid_actions,
        }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out_dir / "train_rollouts.csv", train_rollouts)
    write_csv(out_dir / "online_eval_checkpoints.csv", online_eval)
    write_csv(out_dir / "final_5400_eval.csv", [final_eval])
    (out_dir / "config_fingerprint.json").write_text(
        json.dumps(config_fingerprint(cfg), indent=2), encoding="utf-8"
    )
    print(f"WROTE {out_dir}", flush=True)
    print(json.dumps({
        "algorithm": display,
        "final_5400_eval": final_eval,
        "checkpoint_sha256": checkpoint_sha256,
        "wall_time_sec": elapsed,
    }), flush=True)


if __name__ == "__main__":
    main()
