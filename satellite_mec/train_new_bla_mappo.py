"""Train and evaluate the paper-faithful ``new_BLA-MAPPO`` algorithm.

Examples
--------
Paper-scale run (192 satellites)::

    python train_new_bla_mappo.py --t_train 32000 --n_runs 5

Fast integration run (25 satellites)::

    python train_new_bla_mappo.py --small_scale --debug --no_eval
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from core import SatelliteMECEnv
from evaluation import ExperimentRunner
from new_bla_mappo import NewBLAMAPPOConfig, NewBLAMAPPOPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train the new paper-faithful BLA-MAPPO")
    parser.add_argument("--t_train", type=int, default=32_000)
    parser.add_argument("--t_eval", type=int, default=5_400)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--rho_l", type=float, default=10.0)
    parser.add_argument("--eta_c", type=float, default=0.1)
    parser.add_argument("--small_scale", action="store_true",
                        help="Use the repository's 25-satellite topology for smoke tests")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint to load before training/evaluation")
    parser.add_argument("--model_name", default="new_BLA-MAPPO",
                        help="Directory name used for the exported final model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Root seed for network, tasks, links and initial DoD")
    return parser.parse_args()


def evaluate(policy, env, cfg, n_runs: int, t_eval: int) -> list[dict]:
    rows = []
    policy.set_eval_mode()
    for run_idx in range(n_runs):
        # Paper protocol: warm up one full orbit, then start formal metrics
        # from the same system snapshot without resetting queues/batteries.
        env.reset(phase="warmup", seeds=cfg.get_eval_seeds(run_idx))
        for _ in range(cfg.T_WARMUP):
            policy.run_step(env)
        env.begin_evaluation_from_current_state()
        cumulative_lifetime = 0.0
        energy, delays, slot_lifetime = [], [], []
        last_info = {}
        start = time.time()
        for _ in range(t_eval):
            _, _, last_info = policy.run_step(env)
            slot_hl = float(sum(last_info["paper_lifetime_losses"]))
            cumulative_lifetime += slot_hl
            slot_lifetime.append(slot_hl)
            energy.append(last_info["slot_system_energy"])
            delays.extend(last_info["slot_e2e_delays"])
        hl_tail_size = min(640, len(slot_lifetime))
        rows.append({
            "run": run_idx,
            "completion_rate": last_info.get("eval_completion_rate", 0.0),
            "satisfaction": last_info.get("eval_satisfaction_rate", 0.0),
            "cumulative_lifetime_loss": cumulative_lifetime,
            "mean_lifetime_loss_per_slot": float(np.mean(slot_lifetime)),
            "last_640_mean_lifetime_loss": float(
                np.mean(slot_lifetime[-hl_tail_size:])),
            "last_640_slots_used": hl_tail_size,
            "mean_system_energy_j": float(np.mean(energy)),
            "mean_completed_delay_s": float(np.mean(delays)) if delays else 0.0,
            "elapsed_s": time.time() - start,
        })
    return rows


def episode_mean_rewards(reward_curve: np.ndarray, episode_len: int) -> np.ndarray:
    """Aggregate the complete slots into the paper's fixed-length episodes."""
    n_episode = reward_curve.size // episode_len
    if n_episode == 0:
        return np.asarray([float(np.mean(reward_curve))], dtype=np.float64)
    return reward_curve[:n_episode * episode_len].reshape(n_episode, episode_len).mean(axis=1)


def analyze_plateau(episode_rewards: np.ndarray, tail: int = 30) -> dict:
    """Pre-registered plateau test on the last episodes, not visual judgment."""
    n = min(tail, episode_rewards.size)
    values = np.asarray(episode_rewards[-n:], dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, values, 1) if n >= 2 else (0.0, float(values[-1]))
    fitted = slope * x + intercept
    if n > 2:
        residual_var = float(np.sum((values - fitted) ** 2) / (n - 2))
        slope_se = float(np.sqrt(residual_var / max(np.sum((x - x.mean()) ** 2), 1e-12)))
    else:
        slope_se = float("inf")
    half = n // 2
    window_change = float(np.mean(values[-half:]) - np.mean(values[:half])) if half else 0.0
    ci95 = 1.96 * slope_se
    # Exact-flat synthetic tails can yield machine-epsilon slopes with an even
    # smaller fitted CI; treat sub-1e-12 slopes as numerically zero.
    significant_trend = bool(abs(slope) > max(ci95, 1e-12))
    # The effect-size guard prevents calling a visibly drifting but noisy tail
    # a plateau merely because its regression lacks power.
    plateau = bool(n >= 30 and not significant_trend and abs(window_change) <= 0.5)
    return {
        "tail_episodes": n,
        "tail_mean_reward": float(np.mean(values)),
        "tail_std_reward": float(np.std(values)),
        "slope_reward_per_episode": float(slope),
        "slope_95pct_half_width": float(ci95),
        "second_half_minus_first_half": window_change,
        "criteria": "95% slope CI includes zero and |half-window change| <= 0.5",
        "plateau_detected": plateau,
    }


def plot_reward_curve(reward_curve: np.ndarray, episode_len: int,
                      output_path: str) -> tuple[np.ndarray, dict]:
    """Plot episode mean reward and a 10-episode moving mean."""
    episode_rewards = episode_mean_rewards(reward_curve, episode_len)
    plateau = analyze_plateau(episode_rewards)
    x = np.arange(1, episode_rewards.size + 1)
    window = min(10, episode_rewards.size)
    moving = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, episode_rewards, color="#9E9E9E", marker="o", markersize=2.5,
            alpha=0.55, linewidth=0.8, label=f"Episode mean ({episode_len} slots)")
    ax.plot(x[window - 1:], moving, color="#2CA02C", linewidth=2.2,
            label=f"{window}-episode moving mean")
    tail = plateau["tail_episodes"]
    ax.axvspan(max(1, len(x) - tail + 1), len(x), color="#B0BEC5", alpha=0.14,
               label="Plateau test window")
    status = "plateau detected" if plateau["plateau_detected"] else "still trending"
    ax.set(title=f"new_BLA-MAPPO Training Reward (5x5, {status})",
           xlabel="Training episode", ylabel="Mean system reward per slot")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return episode_rewards, plateau


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = NewBLAMAPPOConfig(small_scale=args.small_scale)
    cfg.SEED = args.seed
    cfg.SEED_TASK = args.seed
    cfg.SEED_TASK_PARAM = args.seed + 1
    cfg.SEED_LINK = args.seed + 2
    cfg.SEED_NET = args.seed + 3
    cfg.SEED_TRAIN = args.seed + 4
    cfg.T_TRAIN = args.t_train
    cfg.T_EVAL = args.t_eval
    cfg.T_TOTAL = max(cfg.T_TRAIN, cfg.T_WARMUP + cfg.T_EVAL) + cfg.ORBIT_PERIOD
    cfg.N_EVAL_RUNS = args.n_runs
    cfg.BLA_RHO_L = args.rho_l
    cfg.BLA_ETA_C = args.eta_c

    runner = ExperimentRunner(cfg, "new_BLA_MAPPO", debug=args.debug)
    runner.setup_algorithm_dir("new_BLA-MAPPO")
    env = SatelliteMECEnv(cfg)
    policy = NewBLAMAPPOPolicy(cfg)
    if args.checkpoint:
        policy.load(args.checkpoint)

    policy.calibrate_normalizers(env)
    runner.run_training(policy, env, seeds={
        "task": cfg.SEED_TASK,
        "task_param": cfg.SEED_TASK_PARAM,
        "dod_init": cfg.SEED,
    })
    source_model_dir = os.path.join(runner.result_dirs[policy.name], "model")
    exported_model_dir = os.path.join(runner.base_dir, args.model_name)
    if os.path.abspath(source_model_dir) != os.path.abspath(exported_model_dir):
        shutil.copytree(source_model_dir, exported_model_dir)
        print(f"Final model exported to {exported_model_dir}")
    if args.no_eval:
        return

    rows = evaluate(policy, env, cfg, cfg.N_EVAL_RUNS, cfg.T_EVAL)
    output_path = os.path.join(runner.base_dir, "new_bla_eval.json")
    reward_path = os.path.join(runner.result_dirs[policy.name], "reward_curve.json")
    with open(reward_path, "r", encoding="utf-8") as stream:
        reward_curve = np.asarray(json.load(stream), dtype=np.float64)
    reward_plot_path = os.path.join(runner.base_dir, "training_reward_curve.png")
    episode_rewards, plateau = plot_reward_curve(
        reward_curve, cfg.K_ROLLOUT, reward_plot_path)
    episode_reward_path = os.path.join(
        runner.result_dirs[policy.name], "episode_reward_curve.json")
    with open(episode_reward_path, "w", encoding="utf-8") as stream:
        json.dump(episode_rewards.tolist(), stream, indent=2)
    tail_size = min(100, reward_curve.size)
    summary = {
        "algorithm": policy.name,
        "saved_model": args.model_name,
        "paper_equations": [10, 12, 18, 21, 22, 23, 25, 26, 27, 28, 29,
                            31, 32, 33, 34, 35, 36, 37, 38],
        "config": {
            "n_sats": cfg.N_SATS,
            "seed": args.seed,
            "t_train": cfg.T_TRAIN,
            "t_eval": cfg.T_EVAL,
            "rho_l": cfg.BLA_RHO_L,
            "lambda_q": cfg.BLA_LAMBDA_Q,
            "lambda_l": cfg.BLA_LAMBDA_L,
            "eta_c": cfg.BLA_ETA_C,
            "lifetime_norm": policy.slot_lifetime_norm.value,
            "task_lifetime_norm": policy.task_lifetime_norm.value,
        },
        "training_reward": {
            "slots": int(reward_curve.size),
            "mean": float(np.mean(reward_curve)),
            "minimum": float(np.min(reward_curve)),
            "maximum": float(np.max(reward_curve)),
            "first_100_mean": float(np.mean(reward_curve[:tail_size])),
            "last_100_mean": float(np.mean(reward_curve[-tail_size:])),
            "curve_file": os.path.relpath(reward_path, runner.base_dir),
            "plot_file": os.path.basename(reward_plot_path),
            "episode_curve_file": os.path.relpath(episode_reward_path, runner.base_dir),
            "episode_count": int(episode_rewards.size),
            "plateau_analysis": plateau,
        },
        "runs": rows,
    }
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"Evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
