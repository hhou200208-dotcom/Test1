#!/usr/bin/env python3
"""8K reproduction diagnostic for the historical BLA/LyaMAPPO implementation.

This script is intentionally executed with PYTHONPATH pointing at the historical
commit c5d8f909987c77d2a7abc708b9e62f4594c33eec.  It does not import the current
branch's core/training implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from core import Config, SatelliteMECEnv, LyapunovCalculator
from training import MAPPOPolicy

HISTORICAL_COMMIT = "c5d8f909987c77d2a7abc708b9e62f4594c33eec"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--reference", default="")
    p.add_argument("--train-slots", type=int, default=8000)
    p.add_argument("--final-eval-slots", type=int, default=5400)
    p.add_argument("--seed", type=int, default=46)
    return p.parse_args()


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.array([], dtype=float)
    return np.convolve(x, np.ones(w, dtype=float) / w, mode="valid")


def safe_float(x, default=0.0):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def slot_energy(env):
    comp = sum(safe_float(getattr(s, "_comp_energy", 0.0)) for s in env.constellation.satellites)
    trans = sum(safe_float(getattr(s, "_trans_energy", 0.0)) for s in env.constellation.satellites)
    return comp, trans


def slot_system_hl(env):
    return sum(safe_float(getattr(s, "slot_health_loss", 0.0)) for s in env.constellation.satellites)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_cfg(train_slots: int, final_eval_slots: int):
    class ReproConfig(Config):
        LAMBDA_HIGH = 4.0
        LAMBDA = LAMBDA_HIGH * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)

    cfg = ReproConfig()
    cfg.T_TRAIN = int(train_slots)
    cfg.T_EVAL = int(final_eval_slots)
    cfg.N_EVAL_RUNS = 1
    cfg.V = 50.0
    cfg.ETA = 0.5
    cfg.BETA = 0.02
    cfg.W_DONE = 10.0
    cfg.W_TIMEOUT = 5.0
    cfg.W_REJECT = 5.0
    cfg.W_HL = 2.0
    cfg.W_QUEUE = 0.05
    cfg.LR_ACTOR = 1e-4
    cfg.LR_CRITIC = 1e-3
    cfg.GAMMA = 0.99
    cfg.LAMBDA_GAE = 0.95
    cfg.EPSILON = 0.2
    cfg.MINIBATCH = 64
    cfg.EPOCH = 2
    cfg.K_ROLLOUT = 64
    cfg.EVAL_INTERVAL = 15
    return cfg


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = make_cfg(args.train_slots, args.final_eval_slots)
    env = SatelliteMECEnv(cfg)
    env.lyapunov_calc = LyapunovCalculator(cfg)
    policy = MAPPOPolicy(cfg, name="BLA_MAPPO_HISTORICAL")

    manifest = {
        "historical_commit": HISTORICAL_COMMIT,
        "purpose": "historical BLA 8K reproduction diagnostic",
        "seed": args.seed,
        "train_slots": cfg.T_TRAIN,
        "final_eval_slots": cfg.T_EVAL,
        "n_sats": cfg.N_SATS,
        "n_planes": cfg.N_PLANES,
        "n_sats_per_plane": cfg.N_SATS_PER_PLANE,
        "lambda_high": cfg.LAMBDA_HIGH,
        "lambda_low": cfg.LAMBDA_LOW,
        "lambda_high_ratio": cfg.LAMBDA_HIGH_RATIO,
        "lambda_mean": cfg.LAMBDA,
        "V": cfg.V,
        "ETA": cfg.ETA,
        "BETA_entropy": cfg.BETA,
        "LR_ACTOR": cfg.LR_ACTOR,
        "LR_CRITIC": cfg.LR_CRITIC,
        "GAMMA": cfg.GAMMA,
        "LAMBDA_GAE": cfg.LAMBDA_GAE,
        "EPSILON": cfg.EPSILON,
        "MINIBATCH": cfg.MINIBATCH,
        "EPOCH": cfg.EPOCH,
        "K_ROLLOUT": cfg.K_ROLLOUT,
        "EVAL_INTERVAL_updates": cfg.EVAL_INTERVAL,
        "historical_task_credit_beta": 0.5,
        "W_DONE": cfg.W_DONE,
        "W_TIMEOUT": cfg.W_TIMEOUT,
        "W_REJECT": cfg.W_REJECT,
        "W_HL": cfg.W_HL,
        "W_QUEUE": cfg.W_QUEUE,
        "KAPPA": cfg.KAPPA,
        "E_CAP": cfg.E_CAP,
        "DOD_MIN": cfg.DOD_MIN,
        "DOD_MAX": cfg.DOD_MAX,
        "CPU_FREQ": cfg.CPU_FREQ,
        "P_SOLAR_MAX": cfg.P_SOLAR_MAX,
        "P_HOUSEKEEPING": cfg.P_HOUSEKEEPING,
        "note": "Torch/NumPy/Python are explicitly seeded for this diagnostic; the historical runner did not consistently seed torch initialization.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Historical training protocol: continuing 64-slot rollouts, QuickEval every
    # 15 PPO updates. QuickEval itself resets the historical training env after
    # evaluation; this is intentionally preserved because it is part of the old
    # runner behavior that produced the golden lineage.
    env.reset(phase="train")
    policy.set_train_mode()
    raw_reward_sum = []
    raw_reward_mean_agent = []
    train_rows = []
    rollout_rows = []
    ppo_rows = []
    rollout_acc = []
    start = time.time()
    last_update_seen = 0

    for t in range(cfg.T_TRAIN):
        rewards, done, info = policy.run_step(env)
        r_sum = safe_float(sum(rewards.values()))
        r_mean = r_sum / max(cfg.N_SATS, 1)
        comp_e, trans_e = slot_energy(env)
        sys_hl = slot_system_hl(env)
        raw_reward_sum.append(r_sum)
        raw_reward_mean_agent.append(r_mean)
        rollout_acc.append(r_sum)
        train_rows.append({
            "train_slot": t + 1,
            "env_slot": info.get("slot", 0),
            "system_reward_sum": r_sum,
            "mean_agent_reward": r_mean,
            "arrived": info.get("arrived", 0),
            "done_tasks": info.get("done_tasks", 0),
            "slot_satisfied": info.get("slot_satisfied", 0),
            "completion_rate_episode": info.get("completion_rate", 0.0),
            "avg_dod": info.get("avg_dod", 0.0),
            "system_lifetime_loss": sys_hl,
            "comp_energy_j": comp_e,
            "trans_energy_j": trans_e,
            "dynamic_energy_j": comp_e + trans_e,
        })

        if done:
            uc = int(policy.trainer.update_count)
            rollout_rows.append({
                "update": uc,
                "train_slots": t + 1,
                "rollout_reward_sum": float(np.sum(rollout_acc)),
                "rollout_reward_mean_per_slot": float(np.mean(rollout_acc)),
                "rollout_reward_mean_agent_per_slot": float(np.mean(rollout_acc) / cfg.N_SATS),
            })
            rollout_acc = []
            if uc > last_update_seen and policy.trainer.train_logs:
                m = policy.trainer.train_logs[-1]
                ppo_rows.append({
                    "update": uc,
                    "train_slots": t + 1,
                    "actor_loss": safe_float(m.get("actor_loss")),
                    "critic_loss": safe_float(m.get("critic_loss")),
                    "ratio_mean": safe_float(m.get("ratio_mean")),
                    "entropy_mean": safe_float(m.get("entropy_mean")),
                    "buffer_task_size": int(m.get("buffer_task_size", 0)),
                    "buffer_slot_size": int(m.get("buffer_slot_size", 0)),
                })
                last_update_seen = uc

            if uc > 0 and uc % cfg.EVAL_INTERVAL == 0:
                policy.quick_eval(env, n_slots=min(cfg.T_EVAL, 1000))

    train_time = time.time() - start
    ckpt = out / "checkpoint_8k"
    policy.save(str(ckpt), extra_info={"historical_commit": HISTORICAL_COMMIT, "repro_seed": args.seed})

    (out / "reward_curve.json").write_text(json.dumps(raw_reward_sum), encoding="utf-8")
    (out / "reward_curve_mean_agent.json").write_text(json.dumps(raw_reward_mean_agent), encoding="utf-8")
    (out / "quick_eval_curve.json").write_text(json.dumps(policy.learning_curve, indent=2), encoding="utf-8")
    write_csv(out / "train_slots.csv", train_rows)
    write_csv(out / "rollouts.csv", rollout_rows)
    write_csv(out / "ppo_updates.csv", ppo_rows)

    # Final independent deterministic evaluation from the 8K checkpoint.
    policy.set_eval_mode()
    env.reset(phase="eval", seeds=cfg.get_eval_seeds(0))
    ev_reward = []
    total_arrived = total_done = total_satisfied = total_timeout = 0
    total_delay = 0.0
    n_delay = 0
    total_comp_e = total_trans_e = total_hl = 0.0
    eval_dod = []
    for _ in range(cfg.T_EVAL):
        rewards, _, info = policy.run_step(env)
        ev_reward.append(safe_float(sum(rewards.values())))
        total_arrived += int(info.get("arrived", 0))
        total_done += int(info.get("done_tasks", 0))
        total_satisfied += int(info.get("slot_satisfied", 0))
        total_timeout += int(info.get("slot_timeout", 0))
        delays = info.get("slot_e2e_delays", []) or []
        total_delay += float(np.sum(delays)) if delays else 0.0
        n_delay += len(delays)
        comp_e, trans_e = slot_energy(env)
        total_comp_e += comp_e
        total_trans_e += trans_e
        total_hl += slot_system_hl(env)
        eval_dod.append(safe_float(info.get("avg_dod", 0.0)))

    eval_summary = {
        "eval_slots": cfg.T_EVAL,
        "system_reward_sum_total": float(np.sum(ev_reward)),
        "system_reward_sum_per_slot": float(np.mean(ev_reward)),
        "mean_agent_reward_per_slot": float(np.mean(ev_reward) / cfg.N_SATS),
        "arrived": total_arrived,
        "done": total_done,
        "satisfied": total_satisfied,
        "timeout": total_timeout,
        "completion_rate_done_over_arrived": total_done / max(total_arrived, 1),
        "satisfaction_satisfied_over_arrived": total_satisfied / max(total_arrived, 1),
        "historical_satisfaction_satisfied_over_done_plus_timeout": total_satisfied / max(total_done + total_timeout, 1),
        "avg_completed_e2e_delay_s": total_delay / max(n_delay, 1),
        "system_delay_overhead_s_per_slot": total_delay / max(cfg.T_EVAL, 1),
        "total_dynamic_energy_j": total_comp_e + total_trans_e,
        "system_energy_overhead_kj_per_slot": (total_comp_e + total_trans_e) / max(cfg.T_EVAL, 1) / 1000.0,
        "cumulative_system_lifetime_loss": total_hl,
        "avg_dod": float(np.mean(eval_dod)) if eval_dod else 0.0,
        "train_wall_time_sec": train_time,
    }
    (out / "final_eval_5400.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    # Old paper-figure convention: moving average of per-slot *system-sum*
    # reward, followed by x64 to show episode-magnitude reward.
    r = np.asarray(raw_reward_sum, dtype=float)
    sm700 = moving_average(r, 700)
    sm1600 = moving_average(r, 1600)
    diag = {
        "n_reward_slots": len(r),
        "raw_reward_mean": float(np.mean(r)),
        "raw_reward_std": float(np.std(r)),
        "first_quarter_mean": float(np.mean(r[: max(len(r)//4, 1)])),
        "last_quarter_mean": float(np.mean(r[-max(len(r)//4, 1):])),
        "smoothed700_first_episode_magnitude": float(sm700[0] * 64) if len(sm700) else None,
        "smoothed700_last_episode_magnitude": float(sm700[-1] * 64) if len(sm700) else None,
        "smoothed1600_first_episode_magnitude": float(sm1600[0] * 64) if len(sm1600) else None,
        "smoothed1600_last_episode_magnitude": float(sm1600[-1] * 64) if len(sm1600) else None,
        "ppo_entropy_first": ppo_rows[0]["entropy_mean"] if ppo_rows else None,
        "ppo_entropy_last": ppo_rows[-1]["entropy_mean"] if ppo_rows else None,
        "quick_eval_last": policy.learning_curve[-1] if policy.learning_curve else None,
    }

    ref = None
    if args.reference and Path(args.reference).exists():
        ref = np.asarray(json.loads(Path(args.reference).read_text(encoding="utf-8")), dtype=float)[: cfg.T_TRAIN]
        ref_sm = moving_average(ref, 700)
        diag.update({
            "reference_slots_used": int(len(ref)),
            "reference_first_quarter_mean": float(np.mean(ref[: max(len(ref)//4, 1)])),
            "reference_last_quarter_mean": float(np.mean(ref[-max(len(ref)//4, 1):])),
            "reference_smoothed700_first_episode_magnitude": float(ref_sm[0] * 64) if len(ref_sm) else None,
            "reference_smoothed700_last_episode_magnitude": float(ref_sm[-1] * 64) if len(ref_sm) else None,
        })

    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(r), dtype=float) / 64.0
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if len(sm700):
            sx = x[350:350 + len(sm700)]
            ax.plot(sx, sm700 * 64, label="8K reproduction", linewidth=2.0)
        if ref is not None:
            rs = moving_average(ref, 700)
            if len(rs):
                rx = np.arange(len(ref), dtype=float) / 64.0
                rsx = rx[350:350 + len(rs)]
                ax.plot(rsx, rs * 64, label="original curve: first 8K", linewidth=1.6, alpha=0.8)
        ax.set_xlabel("Episode (64 physical slots)")
        ax.set_ylabel("Reward (historical episode-magnitude convention)")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "reward_reproduction_vs_original.png", dpi=200)
        plt.close(fig)

        if ppo_rows:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot([x["train_slots"] for x in ppo_rows], [x["entropy_mean"] for x in ppo_rows], linewidth=1.8)
            ax.set_xlabel("Training physical slots")
            ax.set_ylabel("PPO normalized entropy")
            ax.grid(True, linestyle=":", alpha=0.35)
            fig.tight_layout()
            fig.savefig(out / "ppo_entropy.png", dpi=200)
            plt.close(fig)
    except Exception as e:
        (out / "plot_error.txt").write_text(repr(e), encoding="utf-8")

    # Hash the actor so the exact diagnostic checkpoint can be identified later.
    actor_path = ckpt / "actor.pth"
    if actor_path.exists():
        h = hashlib.sha256(actor_path.read_bytes()).hexdigest()
        (out / "actor_sha256.txt").write_text(h + "\n", encoding="utf-8")

    print(json.dumps({"status": "complete", "output": str(out), "diagnostics": diag, "final_eval": eval_summary}, indent=2))


if __name__ == "__main__":
    main()
