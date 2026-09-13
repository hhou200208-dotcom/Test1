"""Zero-shot business-load robustness evaluation for frozen BLA-MAPPO.

The gold policy was trained at the repository's baseline load
(``LAMBDA_HIGH=4.0``, ``LAMBDA_LOW=0.1``).  Evaluation scales both Poisson
rates by the same factor while retaining the fixed high-load satellite set.
No training or fine-tuning is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from core import Config, SatelliteMECEnv
from run_dod_energy_attribution import AttributionConfig
from training import MAPPOPolicy


HERE = Path(__file__).resolve().parent
GOLD_CHECKPOINT = HERE / "checkpoints" / "LyaMAPPO_lh4_32K"
DEFAULT_ALPHAS = (0.5, 0.75, 1.0, 1.25, 1.5)
BASE_LAMBDA_HIGH = 4.0
BASE_LAMBDA_LOW = 0.1
SUMMARY_FIELDS = (
    "algorithm", "alpha", "lambda_high", "lambda_low", "n_planes",
    "sats_per_plane", "n_sats", "seed", "warmup_slots", "eval_slots",
    "user_satisfaction", "mean_system_delay_per_slot_s",
    "mean_system_energy_per_slot_j", "total_lifetime_loss",
    "arrived_tasks", "completed_tasks", "ontime_tasks", "timeout_tasks",
    "rejected_tasks", "mean_completed_task_delay_s",
)
SLOT_FIELDS = (
    "algorithm", "alpha", "lambda_high", "lambda_low", "seed", "eval_slot",
    "absolute_slot", "slot_arrived_tasks", "slot_completed_tasks",
    "slot_ontime_tasks", "slot_timeout_tasks", "slot_rejected_tasks",
    "system_delay_s", "system_energy_j", "system_lifetime_loss",
    "mean_dod", "max_dod",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_config(alpha: float, seed: int, warmup_slots: int, eval_slots: int) -> Config:
    """Create the native 5x5 config with both arrival-rate tiers scaled."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    lambda_high = BASE_LAMBDA_HIGH * float(alpha)
    lambda_low = BASE_LAMBDA_LOW * float(alpha)
    scaled_type = type(
        f"LoadRobustnessConfig_{str(alpha).replace('.', '_')}",
        (AttributionConfig,),
        {
            "LAMBDA_HIGH": lambda_high,
            "LAMBDA_LOW": lambda_low,
            "LAMBDA": (
                lambda_high * Config.LAMBDA_HIGH_RATIO
                + lambda_low * (1 - Config.LAMBDA_HIGH_RATIO)
            ),
        },
    )
    cfg = scaled_type()
    # Config derives queue admission/normalization constants from LAMBDA_HIGH.
    # Freeze those constants at the alpha=1 training/evaluation values so that
    # arrival intensity is the experiment's only changed variable.
    baseline_cfg = AttributionConfig()
    cfg.LAMBDA_MAX = baseline_cfg.LAMBDA_MAX
    cfg.THETA_NUM = baseline_cfg.THETA_NUM
    cfg.Q_NORM = baseline_cfg.Q_NORM
    cfg.SEED = int(seed)
    cfg.SEED_TASK = seed
    cfg.SEED_TASK_PARAM = seed + 1
    cfg.SEED_LINK = seed + 2
    cfg.SEED_NET = seed + 3
    cfg.SEED_TRAIN = seed + 4
    cfg.T_WARMUP = int(warmup_slots)
    cfg.T_EVAL = int(eval_slots)
    cfg.T_TOTAL = warmup_slots + eval_slots + cfg.ORBIT_PERIOD
    cfg.N_EVAL_RUNS = 1
    # Keep the current gold-checkpoint evaluation physics exactly unchanged.
    cfg.V_DVFS = 2e17
    return cfg


def load_gold_policy(cfg: Config, checkpoint: Path) -> MAPPOPolicy:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    policy = MAPPOPolicy(cfg, name="BLA-MAPPO-load-robustness")
    policy.load(str(checkpoint))
    policy.set_eval_mode()
    return policy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_results(summary_csv: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    with summary_csv.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["alpha"]))
    if not rows:
        raise ValueError(f"no rows in {summary_csv}")

    plots = (
        ("user_satisfaction", "User satisfaction", "User satisfaction", "fig_1_user_satisfaction.png"),
        ("mean_system_delay_per_slot_s", "System time-delay overhead", "Mean total delay per slot (s)", "fig_2_system_delay.png"),
        ("mean_system_energy_per_slot_j", "System energy overhead", "Mean total energy per slot (J)", "fig_3_system_energy.png"),
        ("total_lifetime_loss", "Total lifetime loss", "Total lifetime loss", "fig_4_total_lifetime_loss.png"),
    )
    x = np.asarray([float(row["alpha"]) for row in rows])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def draw(ax, field: str, title: str, ylabel: str) -> None:
        y = np.asarray([float(row[field]) for row in rows])
        ax.plot(x, y, color="#D62728", marker="o", linewidth=2.4, markersize=6)
        ax.axvline(1.0, color="#777777", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xlabel(r"Load scaling factor, $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    for field, title, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        draw(ax, field, f"{title} vs. business load", ylabel)
        ax.text(
            0.01, -0.22,
            "Frozen BLA-MAPPO trained at α=1; 5×5 topology; seed=42 (no CI)",
            transform=ax.transAxes, fontsize=8, color="#555555",
        )
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    for ax, (field, title, ylabel, _) in zip(axes.flat, plots):
        draw(ax, field, title, ylabel)
    fig.suptitle("BLA-MAPPO Business-Load Robustness", fontsize=15, y=0.985)
    fig.text(
        0.5, 0.012,
        "Frozen α=1 BLA-MAPPO; native 5×5 load model; 5,400-slot warm-up + 5,400-slot evaluation; seed=42 (no CI)",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=2.1, w_pad=2.0)
    overview = output_dir / "fig_0_load_robustness_overview.png"
    fig.savefig(overview, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.insert(0, overview)
    return paths


def run_alpha(
    *, alpha: float, seed: int, warmup_slots: int, eval_slots: int,
    checkpoint: Path, slot_writer: csv.DictWriter,
) -> dict:
    seed_everything(seed)
    cfg = make_config(alpha, seed, warmup_slots, eval_slots)
    env = SatelliteMECEnv(cfg)
    policy = load_gold_policy(cfg, checkpoint)
    seeds = {"task": seed + 100, "task_param": seed + 101, "dod_init": seed + 102}
    env.reset(phase="warmup", seeds=seeds)

    for slot in range(warmup_slots):
        env.step(policy=policy)
        if (slot + 1) % 900 == 0 or slot + 1 == warmup_slots:
            print(f"[alpha={alpha:g}] gold warm-up {slot + 1}/{warmup_slots}", flush=True)

    policy.set_eval_mode()
    env.begin_evaluation_from_current_state()
    totals = {
        "arrived": 0, "completed": 0, "ontime": 0, "timeout": 0,
        "rejected": 0, "delay": 0.0, "energy": 0.0, "lifetime": 0.0,
    }
    for eval_slot in range(eval_slots):
        _, _, _, info = env.step(policy=policy)
        system_delay = float(sum(info["slot_e2e_delays"]))
        system_energy = float(info["slot_system_energy"])
        system_lifetime = float(info["avg_health_loss"] * cfg.N_SATS)
        totals["arrived"] += int(info["arrived"])
        totals["completed"] += int(info["done_tasks"])
        totals["ontime"] += int(info["slot_satisfied"])
        totals["timeout"] += int(info["slot_timeout"])
        totals["rejected"] += int(info["rejected"])
        totals["delay"] += system_delay
        totals["energy"] += system_energy
        totals["lifetime"] += system_lifetime
        slot_writer.writerow({
            "algorithm": "BLA-MAPPO", "alpha": alpha,
            "lambda_high": cfg.LAMBDA_HIGH, "lambda_low": cfg.LAMBDA_LOW,
            "seed": seed, "eval_slot": eval_slot, "absolute_slot": info["slot"],
            "slot_arrived_tasks": info["arrived"],
            "slot_completed_tasks": info["done_tasks"],
            "slot_ontime_tasks": info["slot_satisfied"],
            "slot_timeout_tasks": info["slot_timeout"],
            "slot_rejected_tasks": info["rejected"],
            "system_delay_s": system_delay, "system_energy_j": system_energy,
            "system_lifetime_loss": system_lifetime,
            "mean_dod": info["avg_dod"], "max_dod": info["max_dod"],
        })
        if (eval_slot + 1) % 900 == 0 or eval_slot + 1 == eval_slots:
            print(f"[alpha={alpha:g}] formal eval {eval_slot + 1}/{eval_slots}", flush=True)

    denominator = totals["completed"] + totals["timeout"]
    return {
        "algorithm": "BLA-MAPPO", "alpha": alpha,
        "lambda_high": cfg.LAMBDA_HIGH, "lambda_low": cfg.LAMBDA_LOW,
        "n_planes": cfg.N_PLANES, "sats_per_plane": cfg.N_SATS_PER_PLANE,
        "n_sats": cfg.N_SATS, "seed": seed, "warmup_slots": warmup_slots,
        "eval_slots": eval_slots,
        "user_satisfaction": totals["ontime"] / max(denominator, 1),
        "mean_system_delay_per_slot_s": totals["delay"] / eval_slots,
        "mean_system_energy_per_slot_j": totals["energy"] / eval_slots,
        "total_lifetime_loss": totals["lifetime"],
        "arrived_tasks": totals["arrived"], "completed_tasks": totals["completed"],
        "ontime_tasks": totals["ontime"], "timeout_tasks": totals["timeout"],
        "rejected_tasks": totals["rejected"],
        "mean_completed_task_delay_s": totals["delay"] / max(totals["completed"], 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-slots", type=int, default=5_400)
    parser.add_argument("--eval-slots", type=int, default=5_400)
    parser.add_argument("--checkpoint", type=Path, default=GOLD_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "results" / "load_robustness_seed42",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("alphas must be unique")
    started = time.time()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "load_robustness_summary.csv"
    slot_csv = output_dir / "load_robustness_slot_metrics.csv"

    summaries = []
    with slot_csv.open("w", newline="", encoding="utf-8") as stream:
        slot_writer = csv.DictWriter(stream, fieldnames=SLOT_FIELDS)
        slot_writer.writeheader()
        for alpha in sorted(args.alphas):
            summaries.append(run_alpha(
                alpha=alpha, seed=args.seed, warmup_slots=args.warmup_slots,
                eval_slots=args.eval_slots, checkpoint=checkpoint,
                slot_writer=slot_writer,
            ))

    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    figure_paths: Iterable[Path] = []
    if not args.skip_plots:
        figure_paths = plot_results(summary_csv, output_dir / "figures")
    manifest = {
        "experiment": "zero-shot BLA-MAPPO business-load robustness",
        "algorithm_changed": False,
        "retraining_or_finetuning": False,
        "training_alpha": 1.0,
        "checkpoint": str(checkpoint.relative_to(HERE)),
        "actor_sha256": sha256_file(checkpoint / "actor.pth"),
        "native_load_model": "fixed 20% high-load satellites; independent Poisson arrivals",
        "load_scaling": "both LAMBDA_HIGH=4.0 and LAMBDA_LOW=0.1 multiplied by alpha",
        "high_load_satellite_positions_preserved": True,
        "alpha_1_queue_thresholds_and_normalization_preserved": True,
        "alphas": sorted(args.alphas), "seed": args.seed,
        "topology": "5x5 (25 satellites)",
        "warmup_policy": "gold BLA-MAPPO", "warmup_slots_per_alpha": args.warmup_slots,
        "eval_slots_per_alpha": args.eval_slots,
        "metric_definitions": {
            "user_satisfaction": "ontime / (completed + timeout)",
            "system_delay_overhead": "mean over slots of sum of completed-task E2E delays in that slot",
            "system_energy_overhead": "mean over slots of total compute plus transmission energy; housekeeping excluded",
            "total_lifetime_loss": "sum of per-satellite convex lifetime loss over all formal-evaluation slots",
        },
        "single_seed_no_confidence_interval": True,
        "slot_csv_rows": len(args.alphas) * args.eval_slots,
        "figures": [path.name for path in figure_paths],
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"summary CSV: {summary_csv}", flush=True)
    print(f"slot CSV: {slot_csv}", flush=True)
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
