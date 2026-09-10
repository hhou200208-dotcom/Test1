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
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

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
    return parser.parse_args()


def evaluate(policy, env, cfg, n_runs: int, t_eval: int) -> list[dict]:
    rows = []
    policy.set_eval_mode()
    for run_idx in range(n_runs):
        env.reset(phase="eval", seeds=cfg.get_eval_seeds(run_idx))
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


def plot_reward_curve(reward_curve: np.ndarray, output_path: str) -> None:
    """Save the complete per-slot reward and a readable 100-slot moving mean."""
    x = np.arange(1, reward_curve.size + 1)
    window = min(100, reward_curve.size)
    kernel = np.ones(window, dtype=np.float64) / window
    moving = np.convolve(reward_curve, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, reward_curve, color="#4C78A8", alpha=0.22, linewidth=0.7,
            label="Per-slot reward")
    ax.plot(x[window - 1:], moving, color="#E45756", linewidth=2.0,
            label=f"{window}-slot moving mean")
    ax.set(title="new_BLA-MAPPO Training Reward (5x5)",
           xlabel="Training slot", ylabel="System reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = NewBLAMAPPOConfig(small_scale=args.small_scale)
    cfg.T_TRAIN = args.t_train
    cfg.T_EVAL = args.t_eval
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
    runner.run_training(policy, env)
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
    plot_reward_curve(reward_curve, reward_plot_path)
    tail_size = min(100, reward_curve.size)
    summary = {
        "algorithm": policy.name,
        "saved_model": args.model_name,
        "paper_equations": [10, 12, 18, 21, 22, 23, 25, 26, 27, 28, 29,
                            31, 32, 33, 34, 35, 36, 37, 38],
        "config": {
            "n_sats": cfg.N_SATS,
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
        },
        "runs": rows,
    }
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"Evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
