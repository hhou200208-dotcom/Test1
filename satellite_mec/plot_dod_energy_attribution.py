"""Derive metrics and four DoD/energy figures directly from the slot CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("w/o DoD", "w/ Linear-DoD", "BLA-MAPPO")
COLORS = {"w/o DoD": "#7F7F7F", "w/ Linear-DoD": "#1F77B4", "BLA-MAPPO": "#D62728"}
DOD_EDGES = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8), dtype=float)
DOD_LABELS = ("[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8]")
DOD_COLORS = ("#4C78A8", "#72B7B2", "#F2CF5B", "#E45756")
# Pre-registered from the physical maximum: kappa*f_max^3*tau/E_cap is
# approximately 2.22e-4, before the much smaller transmission component.
DISCHARGE_EDGES = np.asarray((0.0, 5.0e-5, 1.5e-4, 3.0e-4), dtype=float)
DISCHARGE_LABELS = ("[0, 5e-5)", "[5e-5, 1.5e-4)", "[1.5e-4, 3e-4]")
DISCHARGE_COLORS = ("#4C78A8", "#F2CF5B", "#E45756")
HIGH_DOD = 0.6
A_COEF = 0.8
MIN_CELL_ROWS = 100
MIN_CELL_SATS = 5


def lifetime_curve(dod: float) -> float:
    return dod * (10.0 ** (A_COEF * (dod - 1.0)))


def psi(dod_start: float, dod_end: float) -> float:
    return max(lifetime_curve(dod_end) - lifetime_curve(dod_start), 0.0)


def dod_bin(value: float) -> int | None:
    if value < DOD_EDGES[0] - 1e-12 or value > DOD_EDGES[-1] + 1e-12:
        return None
    if math.isclose(value, DOD_EDGES[-1], abs_tol=1e-12):
        return len(DOD_LABELS) - 1
    idx = int(np.searchsorted(DOD_EDGES, value, side="right") - 1)
    return idx if 0 <= idx < len(DOD_LABELS) else None


def discharge_bin(value: float) -> int | None:
    if value < DISCHARGE_EDGES[0] - 1e-12 or value > DISCHARGE_EDGES[-1] + 1e-12:
        return None
    if math.isclose(value, DISCHARGE_EDGES[-1], abs_tol=1e-12):
        return len(DISCHARGE_LABELS) - 1
    idx = int(np.searchsorted(DISCHARGE_EDGES, value, side="right") - 1)
    return idx if 0 <= idx < len(DISCHARGE_LABELS) else None


def empty_stats() -> dict:
    return {
        "rows": 0,
        "task_energy_by_dod": np.zeros(len(DOD_LABELS)),
        "total_task_energy": 0.0,
        "high_task_energy": 0.0,
        "total_task_discharge": 0.0,
        "high_task_discharge": 0.0,
        "dod_values": [],
        "high_dod_rows": 0,
        "arrived": 0,
        "completed": 0,
        "ontime": 0,
        "timeout": 0,
        "rejected": 0,
        "completed_bits": 0.0,
        "delay_sum": 0.0,
        "lifetime_loss": 0.0,
        "negative_discharge_rows": 0,
        "over_discharge_bin_rows": 0,
        "battery_floor_clipped_rows": 0,
        "max_counterfactual_formula_error_j": 0.0,
        "max_actual_transition_error_j": 0.0,
    }


def read_and_aggregate(csv_path: Path) -> tuple[dict, dict, dict]:
    """Single streaming pass over CSV; no side input is used for figures."""
    stats = {name: empty_stats() for name in ALGORITHMS}
    # (algorithm, discharge_bin, dod_bin, satellite) -> [sum(delta_L), count]
    cell_sat = defaultdict(lambda: [0.0, 0])
    seed_values = set()
    scenarios = set()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            algorithm = row["algorithm"]
            if algorithm not in stats:
                raise ValueError(f"unexpected algorithm: {algorithm}")
            seed_values.add(int(row["seed"]))
            scenarios.add(row["scenario"])
            s = stats[algorithm]
            cap = float(row["battery_capacity_j"])
            b_start = float(row["battery_start_j"])
            b_end = float(row["battery_end_j"])
            b_zero = float(row["battery_counterfactual_j"])
            solar_energy = float(row["solar_energy_j"])
            base_energy = float(row["base_energy_j"])
            task_energy = float(row["compute_energy_j"]) + float(row["tx_energy_j"])
            d_start = 1.0 - b_start / cap
            d_end = 1.0 - b_end / cap
            d_zero = 1.0 - b_zero / cap
            delta_d = (b_zero - b_end) / cap
            delta_l = psi(d_start, d_end) - psi(d_start, d_zero)
            k = dod_bin(d_start)

            s["rows"] += 1
            s["total_task_energy"] += task_energy
            s["total_task_discharge"] += delta_d
            s["dod_values"].append(d_start)
            s["arrived"] += int(row["arrived_tasks"])
            s["completed"] += int(row["completed_tasks"])
            s["ontime"] += int(row["ontime_tasks"])
            s["timeout"] += int(row["timeout_tasks"])
            s["rejected"] += int(row["rejected_tasks"])
            s["completed_bits"] += float(row["completed_bits"])
            s["delay_sum"] += float(row["completion_delay_sum_s"])
            s["lifetime_loss"] += delta_l
            expected_zero = min(cap, b_start + solar_energy - base_energy)
            raw_actual = min(cap, b_start + solar_energy - base_energy - task_energy)
            expected_actual = max(cap * (1.0 - DOD_EDGES[-1]), raw_actual)
            s["max_counterfactual_formula_error_j"] = max(
                s["max_counterfactual_formula_error_j"], abs(b_zero - expected_zero)
            )
            s["max_actual_transition_error_j"] = max(
                s["max_actual_transition_error_j"], abs(b_end - expected_actual)
            )
            if raw_actual < cap * (1.0 - DOD_EDGES[-1]) - 1e-9:
                s["battery_floor_clipped_rows"] += 1
            if delta_d < -1e-12:
                s["negative_discharge_rows"] += 1
            if k is not None:
                s["task_energy_by_dod"][k] += task_energy
            if d_start >= HIGH_DOD:
                s["high_dod_rows"] += 1
                s["high_task_energy"] += task_energy
                s["high_task_discharge"] += delta_d

            j = discharge_bin(delta_d)
            if j is None:
                if delta_d > DISCHARGE_EDGES[-1]:
                    s["over_discharge_bin_rows"] += 1
                continue
            if k is not None:
                sat = int(row["satellite_id"])
                acc = cell_sat[(algorithm, j, k, sat)]
                acc[0] += delta_l
                acc[1] += 1

    if len(seed_values) != 1:
        raise ValueError(f"expected exactly one seed, got {sorted(seed_values)}")
    if len(scenarios) != 1:
        raise ValueError(f"expected exactly one scenario, got {sorted(scenarios)}")
    return stats, cell_sat, {"seed": next(iter(seed_values)), "scenario": next(iter(scenarios))}


def finalize_summary(stats: dict, meta: dict) -> dict:
    out = {
        "seed": meta["seed"],
        "scenario": meta["scenario"],
        "cross_seed_ci_available": False,
        "ci_note": "One seed only; Fig. c intervals are descriptive satellite-cluster 95% CIs.",
        "fixed_dod_bins": DOD_LABELS,
        "fixed_high_dod_threshold": HIGH_DOD,
        "pre_registered_task_discharge_bins": DISCHARGE_LABELS,
        "algorithms": {},
    }
    for algorithm, s in stats.items():
        completed = s["completed"]
        denom = completed + s["timeout"]
        dods = np.asarray(s["dod_values"], dtype=float)
        out["algorithms"][algorithm] = {
            "satellite_slot_rows": s["rows"],
            "arrived_tasks": s["arrived"],
            "completed_tasks": completed,
            "ontime_tasks": s["ontime"],
            "timeout_tasks": s["timeout"],
            "rejected_tasks": s["rejected"],
            "completion_rate": completed / max(s["arrived"], 1),
            "satisfaction": s["ontime"] / max(denom, 1),
            "mean_completed_delay_s": s["delay_sum"] / max(completed, 1),
            "completed_bits": s["completed_bits"],
            "total_task_energy_j": s["total_task_energy"],
            "task_energy_per_completed_task_j": s["total_task_energy"] / max(completed, 1),
            "task_energy_per_completed_task_by_dod_j": (
                s["task_energy_by_dod"] / max(completed, 1)
            ).tolist(),
            "high_dod_energy_ratio": s["high_task_energy"] / max(s["total_task_energy"], 1e-15),
            "high_dod_discharge_ratio": (
                s["high_task_discharge"] / max(s["total_task_discharge"], 1e-15)
            ),
            "high_dod_occupancy": s["high_dod_rows"] / max(s["rows"], 1),
            "dod_p50": float(np.quantile(dods, 0.50)),
            "dod_p90": float(np.quantile(dods, 0.90)),
            "dod_p95": float(np.quantile(dods, 0.95)),
            "dod_max": float(np.max(dods)),
            "task_induced_lifetime_loss": s["lifetime_loss"],
            "lifetime_loss_per_completed_task": s["lifetime_loss"] / max(completed, 1),
            "high_dod_satellite_slot_count": s["high_dod_rows"],
            "negative_task_discharge_rows": s["negative_discharge_rows"],
            "rows_above_registered_discharge_bins": s["over_discharge_bin_rows"],
            "battery_floor_clipped_rows": s["battery_floor_clipped_rows"],
            "max_counterfactual_formula_error_j": s["max_counterfactual_formula_error_j"],
            "max_actual_transition_error_j": s["max_actual_transition_error_j"],
        }
    bla = out["algorithms"]["BLA-MAPPO"]
    out["comparison_conditions"] = {}
    for baseline in ("w/o DoD", "w/ Linear-DoD"):
        base = out["algorithms"][baseline]
        mean_energy = 0.5 * (
            bla["task_energy_per_completed_task_j"] + base["task_energy_per_completed_task_j"]
        )
        out["comparison_conditions"][baseline] = {
            "absolute_satisfaction_gap": abs(bla["satisfaction"] - base["satisfaction"]),
            "service_within_3_percentage_points": abs(bla["satisfaction"] - base["satisfaction"]) <= 0.03,
            "relative_energy_per_task_gap": abs(
                bla["task_energy_per_completed_task_j"] - base["task_energy_per_completed_task_j"]
            ) / max(mean_energy, 1e-15),
            "energy_per_task_within_5_percent": abs(
                bla["task_energy_per_completed_task_j"] - base["task_energy_per_completed_task_j"]
            ) / max(mean_energy, 1e-15) <= 0.05,
            "bla_lower_lifetime_loss_per_task": (
                bla["lifetime_loss_per_completed_task"] < base["lifetime_loss_per_completed_task"]
            ),
            "bla_lower_high_dod_energy_ratio": (
                bla["high_dod_energy_ratio"] < base["high_dod_energy_ratio"]
            ),
            "bla_lower_high_dod_occupancy": (
                bla["high_dod_occupancy"] < base["high_dod_occupancy"]
            ),
        }
    return out


def style(ax) -> None:
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_a(summary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    x = np.arange(len(ALGORITHMS))
    bottom = np.zeros(len(ALGORITHMS))
    for k, (label, color) in enumerate(zip(DOD_LABELS, DOD_COLORS)):
        values = np.asarray([
            summary["algorithms"][a]["task_energy_per_completed_task_by_dod_j"][k]
            for a in ALGORITHMS
        ])
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="white", width=0.68, label=label)
        bottom += values
    ax.set_xticks(x, ALGORITHMS)
    ax.set_ylabel("Task energy per completed task (J/task)")
    ax.set_title("(a) Task energy allocation by starting DoD")
    ax.legend(title="Starting DoD", frameon=False, fontsize=8.5)
    ax.text(0.01, -0.17, "One seed; no cross-seed CI", transform=ax.transAxes, fontsize=8.5, color="#555")
    style(ax)
    fig.tight_layout()
    fig.savefig(output / "fig_a_energy_by_dod.png", dpi=200)
    plt.close(fig)


def plot_b(summary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    x = np.arange(len(ALGORITHMS))
    width = 0.34
    energy = [100 * summary["algorithms"][a]["high_dod_energy_ratio"] for a in ALGORITHMS]
    discharge = [100 * summary["algorithms"][a]["high_dod_discharge_ratio"] for a in ALGORITHMS]
    ax.bar(x - width / 2, energy, width, color="#F28E2B", label="High-DoD energy ratio")
    ax.bar(x + width / 2, discharge, width, color="#C44E52", label="High-DoD discharge ratio")
    ax.set_xticks(x, ALGORITHMS)
    ax.set_ylabel("Ratio (%)")
    ax.set_title("(b) Task activity occurring at DoD >= 0.6")
    ax.legend(frameon=False)
    ax.text(0.01, -0.17, "One seed; bars are point estimates", transform=ax.transAxes, fontsize=8.5, color="#555")
    style(ax)
    fig.tight_layout()
    fig.savefig(output / "fig_b_high_dod_ratios.png", dpi=200)
    plt.close(fig)


def cell_estimate(cell_sat: dict, algorithm: str, discharge_idx: int,
                  dod_idx: int) -> tuple[float, float, int, int] | None:
    sat_means = []
    total = 0
    for sat in range(25):
        value_sum, count = cell_sat.get((algorithm, discharge_idx, dod_idx, sat), (0.0, 0))
        total += count
        if count:
            sat_means.append(value_sum / count)
    if total < MIN_CELL_ROWS or len(sat_means) < MIN_CELL_SATS:
        return None
    values = np.asarray(sat_means, dtype=float)
    mean = float(np.mean(values))
    ci = float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, ci, total, len(values)


def plot_c(cell_sat: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharex=True, sharey=True)
    x = 0.5 * (DOD_EDGES[:-1] + DOD_EDGES[1:])
    for ax, algorithm in zip(axes, ALGORITHMS):
        for j, (label, color) in enumerate(zip(DISCHARGE_LABELS, DISCHARGE_COLORS)):
            ys, cis = [], []
            for k in range(len(DOD_LABELS)):
                estimate = cell_estimate(cell_sat, algorithm, j, k)
                ys.append(np.nan if estimate is None else estimate[0])
                cis.append(np.nan if estimate is None else estimate[1])
            ys_arr = np.asarray(ys)
            ci_arr = np.asarray(cis)
            ax.plot(x, ys_arr, marker="o", color=color, label=label)
            ax.fill_between(x, ys_arr - ci_arr, ys_arr + ci_arr, color=color, alpha=0.15)
        ax.set_title(algorithm)
        ax.set_xlabel("Starting DoD")
        style(ax)
    axes[0].set_ylabel("Task-induced lifetime loss per slot")
    axes[-1].legend(title="Task-induced delta DoD", fontsize=7.5, frameon=False)
    fig.suptitle("(c) Lifetime loss at matched task-induced discharge", y=1.02)
    fig.text(
        0.5, -0.02,
        "Cells require >=100 satellite-slots and >=5 satellites; bands are satellite-cluster 95% CIs",
        ha="center", fontsize=8.5, color="#555",
    )
    fig.tight_layout()
    fig.savefig(output / "fig_c_lifetime_by_dod_and_discharge.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_d(summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    x = np.arange(len(ALGORITHMS))
    occupancy = [100 * summary["algorithms"][a]["high_dod_occupancy"] for a in ALGORITHMS]
    axes[0].bar(x, occupancy, color=[COLORS[a] for a in ALGORITHMS], width=0.65)
    axes[0].set_xticks(x, ALGORITHMS, rotation=12)
    axes[0].set_ylabel("Satellite-slot occupancy (%)")
    axes[0].set_title("High-DoD occupancy (DoD >= 0.6)")
    style(axes[0])

    offsets = (-0.18, 0.0, 0.18)
    for offset, quantile, marker in zip(offsets, ("dod_p50", "dod_p90", "dod_p95"), ("o", "s", "^")):
        values = [summary["algorithms"][a][quantile] for a in ALGORITHMS]
        axes[1].scatter(x + offset, values, marker=marker, s=56, label=quantile.replace("dod_", "").upper())
    axes[1].set_xticks(x, ALGORITHMS, rotation=12)
    axes[1].set_ylabel("Starting DoD")
    axes[1].set_title("DoD distribution quantiles")
    axes[1].legend(frameon=False)
    style(axes[1])
    fig.suptitle("(d) High-DoD exposure and DoD distribution")
    fig.text(0.5, -0.01, "One seed; no cross-seed CI", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(output / "fig_d_high_dod_occupancy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats, cell_sat, meta = read_and_aggregate(args.csv)
    summary = finalize_summary(stats, meta)
    (args.output_dir / "derived_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_a(summary, args.output_dir)
    plot_b(summary, args.output_dir)
    plot_c(cell_sat, args.output_dir)
    plot_d(summary, args.output_dir)
    print(f"Derived metrics and four figures -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
