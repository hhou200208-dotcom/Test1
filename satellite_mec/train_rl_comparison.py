"""Fair 2K comparison of new_BLA-MAPPO, MAPPO, IPPO, MADDPG and QMIX."""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import SatelliteMECEnv
from new_bla_mappo import NewBLAMAPPOConfig, NewBLAMAPPOPolicy
from rl_comparison import (
    ComparisonConfig, IPPOConfig, ComparisonMAPPOPolicy,
    ComparisonIPPOPolicy, ComparisonMADDPGPolicy, QMIXPolicy,
)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--t_train", type=int, default=2000)
    p.add_argument("--t_eval", type=int, default=1000)
    p.add_argument("--output", default="results/rl_comparison_2k")
    return p.parse_args()


def moving_mean(x, window=100):
    w = min(window, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid"), w


def plot_curves(curves, path):
    colors = {"new_BLA-MAPPO": "#D62728", "MAPPO": "#1F77B4",
              "IPPO": "#2CA02C", "MADDPG": "#9467BD", "QMIX": "#FF7F0E"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, values in curves.items():
        y, w = moving_mean(np.asarray(values, float))
        ax.plot(np.arange(w, len(values) + 1), y, label=name,
                color=colors[name], linewidth=2.2 if name == "MAPPO" else 1.8)
    ax.set(title="Training Reward Comparison (5x5 satellites, 2K slots)",
           xlabel="Training slot", ylabel="Mean reward per satellite")
    ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def train(policy, env, slots):
    env.reset(phase="train")
    policy.set_train_mode()
    rewards = []
    start = time.time()
    for _ in range(slots):
        _, _, info = policy.run_step(env)
        rewards.append(float(info.get(
            "new_bla_system_reward", info.get("comparison_system_reward", 0.0))))
    return rewards, time.time() - start


def evaluate(policy, env, cfg, slots):
    env.reset(phase="eval", seeds=cfg.get_eval_seeds(0))
    policy.set_eval_mode()
    slot_hl, energy, delays = [], [], []
    last = {}
    start = time.time()
    for _ in range(slots):
        _, _, last = policy.run_step(env)
        slot_hl.append(float(sum(last.get("paper_lifetime_losses") or [0.0])))
        energy.append(float(last["slot_system_energy"]))
        delays.extend(last["slot_e2e_delays"])
    tail = min(640, len(slot_hl))
    return {
        "completion_rate": float(last.get("eval_completion_rate", 0.0)),
        "satisfaction": float(last.get("eval_satisfaction_rate", 0.0)),
        "mean_completed_delay_s": float(np.mean(delays)) if delays else 0.0,
        "system_total_completed_delay_s": float(np.sum(delays)),
        "mean_compute_tx_energy_j_per_slot": float(np.mean(energy)),
        "system_total_compute_tx_energy_j": float(np.sum(energy)),
        "cumulative_lifetime_loss": float(np.sum(slot_hl)),
        "last_640_mean_lifetime_loss": float(np.mean(slot_hl[-tail:])),
        "last_640_slots_used": tail,
        "elapsed_s": time.time() - start,
    }


def main():
    args = args_parser()
    os.makedirs(args.output, exist_ok=True)
    torch.set_num_threads(max(min(os.cpu_count() or 1, 4), 1))

    new_cfg = NewBLAMAPPOConfig(small_scale=True)
    calibrator = NewBLAMAPPOPolicy(new_cfg)
    calibrator.calibrate_normalizers(SatelliteMECEnv(new_cfg))
    shared_hl_norm = calibrator.slot_lifetime_norm.value

    def make_new():
        cfg = NewBLAMAPPOConfig(small_scale=True)
        policy = NewBLAMAPPOPolicy(cfg)
        policy.slot_lifetime_norm.frozen_value = shared_hl_norm
        policy.task_lifetime_norm.frozen_value = calibrator.task_lifetime_norm.value
        return cfg, SatelliteMECEnv(cfg), policy

    def make_mappo():
        cfg = ComparisonConfig(); cfg.COMPARISON_HL_NORM = shared_hl_norm
        return cfg, SatelliteMECEnv(cfg), ComparisonMAPPOPolicy(cfg)

    def make_ippo():
        cfg = IPPOConfig(); cfg.COMPARISON_HL_NORM = shared_hl_norm
        return cfg, SatelliteMECEnv(cfg), ComparisonIPPOPolicy(cfg)

    def make_maddpg():
        cfg = ComparisonConfig(); cfg.COMPARISON_HL_NORM = shared_hl_norm
        env = SatelliteMECEnv(cfg)
        return cfg, env, ComparisonMADDPGPolicy(
            cfg, env, name="MADDPG", start_steps=256, batch_size=128,
            replay_size=100_000, anneal_slots=args.t_train, seed=cfg.SEED_TRAIN)

    def make_qmix():
        cfg = ComparisonConfig(); cfg.COMPARISON_HL_NORM = shared_hl_norm
        return cfg, SatelliteMECEnv(cfg), QMIXPolicy(cfg, seed=cfg.SEED_TRAIN)

    factories = {
        "new_BLA-MAPPO": make_new, "MAPPO": make_mappo, "IPPO": make_ippo,
        "MADDPG": make_maddpg, "QMIX": make_qmix,
    }
    curves, summary = {}, {
        "protocol": {"n_sats": 25, "t_train": args.t_train,
                     "t_eval": args.t_eval, "reward": "mean per satellite",
                     "hl_norm": shared_hl_norm},
        "algorithms": {},
    }
    for idx, (name, factory) in enumerate(factories.items()):
        np.random.seed(42); torch.manual_seed(42)
        cfg, env, policy = factory()
        curve, train_s = train(policy, env, args.t_train)
        model_dir = os.path.join(args.output, "models", name)
        policy.save(model_dir, {"t_train": args.t_train})
        metrics = evaluate(policy, env, cfg, args.t_eval)
        curve_path = os.path.join(args.output, f"{name}_reward.json")
        with open(curve_path, "w", encoding="utf-8") as f:
            json.dump(curve, f)
        tail = min(100, len(curve))
        summary["algorithms"][name] = {
            "training": {"elapsed_s": train_s, "mean_reward": float(np.mean(curve)),
                         "first_100_mean": float(np.mean(curve[:tail])),
                         "last_100_mean": float(np.mean(curve[-tail:]))},
            "evaluation": metrics,
        }
        curves[name] = curve
        print(name, summary["algorithms"][name], flush=True)

    plot_curves(curves, os.path.join(args.output, "training_reward_comparison.png"))
    with open(os.path.join(args.output, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
