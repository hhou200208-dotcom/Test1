#!/usr/bin/env python3
"""Validate and plot unified-reward convergence histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np

ALGORITHMS = ("MAPPO", "IPPO", "MADDPG", "TD3")


def load_histories(root: Path) -> Dict[str, Mapping]:
    histories = {}
    for alg in ALGORITHMS:
        path = root / alg.lower() / "history.json"
        if not path.exists():
            raise FileNotFoundError(f"missing required history: {path}")
        with path.open(encoding="utf-8") as f:
            h = json.load(f)
        if h.get("algorithm") != alg:
            raise ValueError(f"algorithm mismatch in {path}")
        if h.get("reward_definition") != "common_reward_v1":
            raise ValueError(f"wrong reward definition in {path}")
        histories[alg] = h
    validate(histories)
    return histories


def validate(histories: Mapping[str, Mapping]) -> None:
    ref = list(histories[ALGORITHMS[0]]["environment_steps"])
    if not ref or ref[0] != 0:
        raise ValueError("evaluation steps must start at step 0")
    for alg in ALGORITHMS:
        h = histories[alg]
        if list(h["environment_steps"]) != ref:
            raise ValueError(f"{alg}: evaluation points differ from MAPPO")
        n = len(ref)
        keys = (
            "gradient_updates", "wall_time_seconds", "eval_return_per_slot",
            "completion_rate", "avg_health_loss", "queue_tasks_per_sat",
            "satisfaction", "avg_delay",
        )
        for key in keys:
            values = np.asarray(h[key], dtype=float)
            if len(values) != n:
                raise ValueError(f"{alg}: {key} length {len(values)} != {n}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{alg}: {key} contains NaN/Inf")
        ledger = h["reward_ledger"]
        for key in ("action_cost", "done", "timeout", "reject", "health_loss", "queue", "total"):
            values = np.asarray(ledger[key], dtype=float)
            if len(values) != n or not np.all(np.isfinite(values)):
                raise ValueError(f"{alg}: invalid ledger series {key}")
    if len(ref) > 1:
        diffs = np.diff(np.asarray(ref, dtype=int))
        # A non-multiple final point is allowed for debug runs; 32K must be exact.
        if ref[-1] == 32000 and not np.array_equal(ref, np.arange(0, 32001, 2000)):
            raise ValueError("32K data must be exactly 0/2K/.../32K")
        if np.any(diffs <= 0):
            raise ValueError("environment steps must be strictly increasing")


def _subtitle(h: Mapping) -> str:
    steps = h["environment_steps"]
    if steps[-1] == 32000:
        return "Preliminary · Single seed (seed=42) · common_reward_v1\n500-slot fixed evaluation every 2,000 environment steps"
    return f"Preliminary · Single seed (seed=42) · common_reward_v1 · run ends at {steps[-1]:,} environment steps"


def plot_metric(histories, root: Path, key: str, ylabel: str, filename: str,
                title: str, xkey: str = "environment_steps") -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for alg in ALGORITHMS:
        h = histories[alg]
        ax.plot(h[xkey], h[key], marker="o", linewidth=1.6, markersize=3.5, label=alg)
    ax.set_xlabel("Environment Steps" if xkey == "environment_steps" else "Training Wall-clock Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{_subtitle(histories['MAPPO'])}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / filename, dpi=180)
    if filename == "reward_convergence.png":
        fig.savefig(root / "reward_convergence.pdf")
    plt.close(fig)


def plot_training_efficiency(histories, root: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for alg in ALGORITHMS:
        h = histories[alg]
        ax.plot(h["wall_time_seconds"], h["eval_return_per_slot"], marker="o",
                linewidth=1.6, markersize=3.5, label=alg)
    ax.set_xlabel("Training Wall-clock Time (s)")
    ax.set_ylabel("Evaluation Return per Slot")
    ax.set_title(f"Training Efficiency\n{_subtitle(histories['MAPPO'])}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "training_efficiency.png", dpi=180)
    plt.close(fig)


def plot_reward_components(histories, root: Path) -> None:
    components = ("action_cost", "done", "timeout", "reject", "health_loss", "queue")
    # One figure per algorithm keeps component scales legible; the four panels are
    # generated as distinct files rather than silently aggregating incompatible axes.
    for alg in ALGORITHMS:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        h = histories[alg]
        for component in components:
            ax.plot(h["environment_steps"], h["reward_ledger"][component],
                    marker="o", markersize=3, linewidth=1.3, label=component)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Reward Component per Evaluation Slot")
        ax.set_title(f"{alg} Reward Ledger\n{_subtitle(histories['MAPPO'])}")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(root / f"reward_components_{alg.lower()}.png", dpi=180)
        plt.close(fig)


def make_plots(root: Path) -> None:
    histories = load_histories(root)
    plot_metric(histories, root, "eval_return_per_slot", "Evaluation Return per Slot",
                "reward_convergence.png", "Unified-Reward Convergence")
    plot_metric(histories, root, "completion_rate", "Completion Rate",
                "completion_rate_convergence.png", "Completion Rate Convergence")
    plot_metric(histories, root, "avg_health_loss", "Average Health Loss",
                "health_loss_convergence.png", "Health-Loss Convergence")
    plot_training_efficiency(histories, root)
    plot_reward_components(histories, root)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", default="results/rl_reward_comparison/seed_42",
                   help="directory containing mappo/ippo/maddpg/td3 history.json files")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    make_plots(root)
    print(f"plots written to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
