"""Zero-shot constellation-size evaluation for five frozen policies.

The repository's existing gold checkpoint was trained with the legacy 5x5
(25-satellite) configuration.  This script does not train or fine-tune it.  It
loads the same frozen learned policies for every Walker size, runs one common
gold BLA-MAPPO warm-up, then forks that exact state for formal evaluation by
BLA-MAPPO, MHSPO, LyDRL-DoD, GDCO, and LSO.

The 16x12 warm-up state is serialized after warm-up and restored before its
formal evaluation.  This makes the experiment itself exercise the saved state
instead of merely writing an unchecked cache file.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from core import Config, SatelliteMECEnv
from baselines import GDCOPolicy, LocalOnlyPolicy, MHSPOPolicy
from baselines.maddpg_dod import MADDPGDoDPolicy
from run_dod_energy_attribution import AttributionConfig, observe_gold_warmup
from training import MAPPOPolicy


HERE = Path(__file__).resolve().parent
GOLD_CHECKPOINT = HERE / "checkpoints" / "LyaMAPPO_lh4_32K"
LYDRL_CHECKPOINT = HERE / "checkpoints" / "MADDPG_DoD_lh4_32K"
DEFAULT_SATS_PER_PLANE = (8, 10, 12, 16, 20)
TRAINING_PLANES = 5
TRAINING_SATS_PER_PLANE = 5
TRAINING_N_SATS = TRAINING_PLANES * TRAINING_SATS_PER_PLANE
SUMMARY_FIELDS = (
    "algorithm", "n_planes", "sats_per_plane", "n_sats", "seed",
    "training_n_planes", "training_sats_per_plane", "training_n_sats",
    "warmup_slots", "eval_slots", "user_satisfaction",
    "mean_system_delay_per_slot_s", "mean_system_energy_per_slot_j",
    "total_lifetime_loss", "arrived_tasks", "completed_tasks",
    "ontime_tasks", "timeout_tasks", "rejected_tasks",
    "mean_completed_task_delay_s",
)
SLOT_FIELDS = (
    "algorithm", "n_planes", "sats_per_plane", "n_sats", "seed", "eval_slot",
    "absolute_slot", "slot_arrived_tasks", "slot_completed_tasks",
    "slot_ontime_tasks", "slot_timeout_tasks", "slot_rejected_tasks",
    "system_delay_s", "system_energy_j", "system_lifetime_loss",
    "mean_dod", "max_dod",
)
ALGORITHMS = ("BLA-MAPPO", "MHSPO", "LyDRL-DoD", "GDCO", "LSO")
PLOT_STYLE = {
    "BLA-MAPPO": ("#D62728", "o"),
    "MHSPO": ("#2CA02C", "^"),
    "LyDRL-DoD": ("#F2C200", "D"),
    "GDCO": ("#9467BD", "P"),
    "LSO": ("#7F7F7F", "*"),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_config(
    planes: int,
    sats_per_plane: int,
    seed: int,
    warmup_slots: int,
    eval_slots: int,
) -> AttributionConfig:
    """Create a pickle-stable config while recomputing all N-dependent fields."""
    cfg = AttributionConfig.__new__(AttributionConfig)
    cfg.N_PLANES = int(planes)
    cfg.N_SATS_PER_PLANE = int(sats_per_plane)
    cfg.N_SATS = cfg.N_PLANES * cfg.N_SATS_PER_PLANE
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
    cfg.LINEAR_DOD_LOSS = False
    Config.__init__(cfg)
    # Preserve the exact evaluation physics used by the current gold-checkpoint
    # mechanism experiments.
    cfg.V_DVFS = 2e17
    return cfg


def load_gold_policy(cfg: Config, checkpoint: Path) -> MAPPOPolicy:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    policy = MAPPOPolicy(cfg, name="BLA-MAPPO-zero-shot-scale")
    policy.load(str(checkpoint))
    policy.set_eval_mode()
    return policy


def make_candidate(
    algorithm: str,
    cfg: Config,
    env: SatelliteMECEnv,
    *,
    seed: int,
    checkpoint: Path,
    lydrl_checkpoint: Path,
    warmed_mhspo_predictors: dict | None,
):
    """Create a fresh evaluation policy bound to its private environment."""
    if algorithm == "BLA-MAPPO":
        return load_gold_policy(cfg, checkpoint)
    if algorithm == "MHSPO":
        policy = MHSPOPolicy(
            cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0, name=algorithm
        )
        if warmed_mhspo_predictors is None:
            raise RuntimeError("MHSPO requires predictor state from the common warm-up")
        policy.predictors = copy.deepcopy(warmed_mhspo_predictors)
        for predictor in policy.predictors.values():
            predictor.cfg = cfg
        return policy
    if algorithm == "LyDRL-DoD":
        if not lydrl_checkpoint.is_dir():
            raise FileNotFoundError(f"missing LyDRL-DoD checkpoint: {lydrl_checkpoint}")
        policy = MADDPGDoDPolicy(cfg, env, seed=seed, name=algorithm)
        policy.load(str(lydrl_checkpoint))
        policy.set_eval_mode()
        return policy
    if algorithm == "GDCO":
        return GDCOPolicy(cfg, env, seed=seed)
    if algorithm == "LSO":
        return LocalOnlyPolicy(cfg, env)
    raise ValueError(f"unknown algorithm: {algorithm}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_warmup_snapshot(
    env: SatelliteMECEnv,
    mhspo_predictors: dict,
    path: Path,
    *,
    seed: int,
    warmup_slots: int,
    checkpoint: Path,
) -> dict:
    """Serialize the complete warmed environment and write an audit sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 2,
        "description": "Gold BLA-MAPPO post-warm-up environment and MHSPO predictor state",
        "n_planes": env.cfg.N_PLANES,
        "sats_per_plane": env.cfg.N_SATS_PER_PLANE,
        "n_sats": env.cfg.N_SATS,
        "seed": seed,
        "warmup_slots": warmup_slots,
        "current_slot": env.current_slot,
        "training_constellation": "5x5 (25 satellites)",
        "source_checkpoint": str(checkpoint.relative_to(HERE)),
        "source_actor_sha256": sha256_file(checkpoint / "actor.pth"),
        "source_commit": os.environ.get("GITHUB_SHA", "local-working-tree"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    raw = pickle.dumps(
        {
            "metadata": metadata,
            "environment": env,
            "online_policy_state": {"MHSPO_predictors": mhspo_predictors},
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    metadata["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    with gzip.open(path, "wb", compresslevel=6) as stream:
        stream.write(raw)
    metadata["compressed_sha256"] = sha256_file(path)
    metadata["compressed_bytes"] = path.stat().st_size
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_warmup_snapshot(path: Path) -> tuple[SatelliteMECEnv, dict, dict]:
    sidecar = path.with_suffix(path.suffix + ".json")
    expected = json.loads(sidecar.read_text(encoding="utf-8"))
    if sha256_file(path) != expected["compressed_sha256"]:
        raise RuntimeError("warm-up snapshot compressed SHA-256 mismatch")
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected["payload_sha256"]:
        raise RuntimeError("warm-up snapshot payload SHA-256 mismatch")
    payload = pickle.loads(raw)
    env = payload["environment"]
    online_policy_state = payload.get("online_policy_state", {})
    predictors = online_policy_state.get("MHSPO_predictors")
    if predictors is None or len(predictors) != env.cfg.N_SATS:
        raise RuntimeError("warm-up snapshot is missing complete MHSPO predictor state")
    embedded = payload["metadata"]
    for key in ("n_planes", "sats_per_plane", "n_sats", "seed", "warmup_slots"):
        if embedded[key] != expected[key]:
            raise RuntimeError(f"warm-up snapshot metadata mismatch for {key}")
    if env.current_slot != expected["current_slot"]:
        raise RuntimeError("restored environment slot mismatch")
    if env.cfg.N_SATS != expected["n_sats"]:
        raise RuntimeError("restored environment satellite count mismatch")
    return env, predictors, expected


def clone_warm_state(env: SatelliteMECEnv, predictors: dict) -> tuple[SatelliteMECEnv, dict]:
    """Fork a byte-identical environment and online predictor state."""
    return pickle.loads(pickle.dumps((env, predictors), protocol=pickle.HIGHEST_PROTOCOL))


def validate_topology(env: SatelliteMECEnv) -> None:
    matrix = env.constellation.neighbor_matrix
    if matrix.shape != (env.cfg.N_SATS, env.cfg.N_NEIGHBORS):
        raise RuntimeError(f"unexpected neighbor matrix shape: {matrix.shape}")
    for sat_id, neighbors in enumerate(matrix):
        if len(set(map(int, neighbors))) != env.cfg.N_NEIGHBORS:
            raise RuntimeError(f"satellite {sat_id} does not have four unique neighbors")
        if sat_id in neighbors:
            raise RuntimeError(f"satellite {sat_id} is its own neighbor")
        for neighbor in neighbors:
            if sat_id not in matrix[int(neighbor)]:
                raise RuntimeError(f"asymmetric topology edge {sat_id}<->{neighbor}")


def plot_results(summary_csv: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    with summary_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["n_sats"]))
    plots = (
        ("user_satisfaction", "User satisfaction", "User satisfaction", "fig_1_user_satisfaction.png"),
        ("mean_system_delay_per_slot_s", "System time-delay overhead", "Mean total delay per slot (s)", "fig_2_system_delay.png"),
        ("mean_system_energy_per_slot_j", "System energy overhead", "Mean total energy per slot (J)", "fig_3_system_energy.png"),
        ("total_lifetime_loss", "Total lifetime loss", "Total lifetime loss", "fig_4_total_lifetime_loss.png"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for field, title, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for algorithm in ALGORITHMS:
            subset = [row for row in rows if row["algorithm"] == algorithm]
            subset.sort(key=lambda row: int(row["n_sats"]))
            if not subset:
                continue
            x = np.asarray([int(row["n_sats"]) for row in subset])
            y = np.asarray([float(row[field]) for row in subset])
            color, marker = PLOT_STYLE[algorithm]
            ax.plot(
                x, y, color=color, marker=marker,
                linewidth=2.4 if algorithm == "BLA-MAPPO" else 1.7,
                markersize=6, label=algorithm,
            )
        expected_x = sorted({int(row["n_sats"]) for row in rows})
        ax.set_xticks(expected_x)
        ax.set_xlabel("Number of satellites, N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} vs. constellation size")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=8.5)
        ax.text(
            0.01, -0.22,
            "Frozen 25-satellite-trained learned policies; common gold warm-up; seed=42 (no CI)",
            transform=ax.transAxes, fontsize=8, color="#555555",
        )
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planes", type=int, default=16)
    parser.add_argument(
        "--sats-per-plane", type=int, nargs="+",
        default=list(DEFAULT_SATS_PER_PLANE),
    )
    parser.add_argument("--warmup-slots", type=int, default=5_400)
    parser.add_argument("--eval-slots", type=int, default=5_400)
    parser.add_argument("--checkpoint", type=Path, default=GOLD_CHECKPOINT)
    parser.add_argument("--lydrl-checkpoint", type=Path, default=LYDRL_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "results" / "scalability_seed42",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path,
        default=HERE / "checkpoints" / "warmup_states",
    )
    parser.add_argument(
        "--reuse-n192-snapshot", action="store_true",
        help="Skip the 16x12 warm-up if a verified saved snapshot exists.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run_scale(
    *,
    planes: int,
    sats_per_plane: int,
    seed: int,
    warmup_slots: int,
    eval_slots: int,
    checkpoint: Path,
    lydrl_checkpoint: Path,
    snapshot_dir: Path,
    reuse_n192_snapshot: bool,
    slot_writer: csv.DictWriter,
) -> tuple[list[dict], dict | None]:
    n_sats = planes * sats_per_plane
    snapshot_path = snapshot_dir / f"bla_mappo_n{n_sats}_seed{seed}_w{warmup_slots}.pkl.gz"
    is_persistent_snapshot = planes == 16 and sats_per_plane == 12
    seed_everything(seed)

    snapshot_metadata = None
    if is_persistent_snapshot and reuse_n192_snapshot and snapshot_path.is_file():
        env, warmed_predictors, snapshot_metadata = load_warmup_snapshot(snapshot_path)
        cfg = env.cfg
        if (cfg.N_PLANES, cfg.N_SATS_PER_PLANE, cfg.N_SATS) != (planes, sats_per_plane, n_sats):
            raise RuntimeError("saved 192-satellite snapshot has incompatible topology")
        cfg.T_WARMUP = int(warmup_slots)
        cfg.T_EVAL = int(eval_slots)
        cfg.T_TOTAL = warmup_slots + eval_slots + cfg.ORBIT_PERIOD
        print(f"[N={n_sats}] restored verified warm-up snapshot {snapshot_path}", flush=True)
    else:
        cfg = make_config(planes, sats_per_plane, seed, warmup_slots, eval_slots)
        env = SatelliteMECEnv(cfg)
        validate_topology(env)
        warmup_policy = load_gold_policy(cfg, checkpoint)
        warmup_mhspo = MHSPOPolicy(
            cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0, name="MHSPO"
        )
        eval_seeds = {
            "task": seed + 100,
            "task_param": seed + 101,
            "dod_init": seed + 102,
        }
        env.reset(phase="warmup", seeds=eval_seeds)
        for slot in range(warmup_slots):
            observe_gold_warmup(warmup_mhspo, env)
            env.step(policy=warmup_policy)
            if (slot + 1) % 900 == 0 or slot + 1 == warmup_slots:
                print(f"[N={n_sats}] gold warm-up {slot + 1}/{warmup_slots}", flush=True)
        warmed_predictors = warmup_mhspo.predictors
        if is_persistent_snapshot:
            snapshot_metadata = save_warmup_snapshot(
                env, warmed_predictors, snapshot_path,
                seed=seed, warmup_slots=warmup_slots,
                checkpoint=checkpoint,
            )
            # Formal evaluation deliberately uses the deserialized object.
            env, warmed_predictors, verified = load_warmup_snapshot(snapshot_path)
            if verified["payload_sha256"] != snapshot_metadata["payload_sha256"]:
                raise RuntimeError("snapshot verification returned a different payload")
            print(f"[N={n_sats}] snapshot round-trip verified", flush=True)

    summaries = []
    for algorithm in ALGORITHMS:
        seed_everything(seed)
        candidate_env, candidate_predictors = clone_warm_state(env, warmed_predictors)
        validate_topology(candidate_env)
        policy = make_candidate(
            algorithm, candidate_env.cfg, candidate_env, seed=seed,
            checkpoint=checkpoint, lydrl_checkpoint=lydrl_checkpoint,
            warmed_mhspo_predictors=candidate_predictors,
        )
        if hasattr(policy, "set_eval_mode"):
            policy.set_eval_mode()
        candidate_env.begin_evaluation_from_current_state()

        totals = {
            "arrived": 0, "completed": 0, "ontime": 0, "timeout": 0,
            "rejected": 0, "delay": 0.0, "energy": 0.0, "lifetime": 0.0,
        }
        for eval_slot in range(eval_slots):
            _, _, _, info = candidate_env.step(policy=policy)
            system_delay = float(sum(info["slot_e2e_delays"]))
            system_energy = float(info["slot_system_energy"])
            system_lifetime = float(info["avg_health_loss"] * candidate_env.cfg.N_SATS)
            totals["arrived"] += int(info["arrived"])
            totals["completed"] += int(info["done_tasks"])
            totals["ontime"] += int(info["slot_satisfied"])
            totals["timeout"] += int(info["slot_timeout"])
            totals["rejected"] += int(info["rejected"])
            totals["delay"] += system_delay
            totals["energy"] += system_energy
            totals["lifetime"] += system_lifetime
            slot_writer.writerow({
                "algorithm": algorithm,
                "n_planes": planes,
                "sats_per_plane": sats_per_plane,
                "n_sats": n_sats,
                "seed": seed,
                "eval_slot": eval_slot,
                "absolute_slot": info["slot"],
                "slot_arrived_tasks": info["arrived"],
                "slot_completed_tasks": info["done_tasks"],
                "slot_ontime_tasks": info["slot_satisfied"],
                "slot_timeout_tasks": info["slot_timeout"],
                "slot_rejected_tasks": info["rejected"],
                "system_delay_s": system_delay,
                "system_energy_j": system_energy,
                "system_lifetime_loss": system_lifetime,
                "mean_dod": info["avg_dod"],
                "max_dod": info["max_dod"],
            })
            if (eval_slot + 1) % 900 == 0 or eval_slot + 1 == eval_slots:
                print(
                    f"[N={n_sats}][{algorithm}] formal eval "
                    f"{eval_slot + 1}/{eval_slots}", flush=True,
                )

        denominator = totals["completed"] + totals["timeout"]
        summaries.append({
            "algorithm": algorithm,
            "n_planes": planes,
            "sats_per_plane": sats_per_plane,
            "n_sats": n_sats,
            "seed": seed,
            "training_n_planes": TRAINING_PLANES,
            "training_sats_per_plane": TRAINING_SATS_PER_PLANE,
            "training_n_sats": TRAINING_N_SATS,
            "warmup_slots": warmup_slots,
            "eval_slots": eval_slots,
            "user_satisfaction": totals["ontime"] / max(denominator, 1),
            "mean_system_delay_per_slot_s": totals["delay"] / eval_slots,
            "mean_system_energy_per_slot_j": totals["energy"] / eval_slots,
            "total_lifetime_loss": totals["lifetime"],
            "arrived_tasks": totals["arrived"],
            "completed_tasks": totals["completed"],
            "ontime_tasks": totals["ontime"],
            "timeout_tasks": totals["timeout"],
            "rejected_tasks": totals["rejected"],
            "mean_completed_task_delay_s": totals["delay"] / max(totals["completed"], 1),
        })
    return summaries, snapshot_metadata


def main() -> None:
    args = parse_args()
    started = time.time()
    checkpoint = args.checkpoint.resolve()
    lydrl_checkpoint = args.lydrl_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    snapshot_dir = args.snapshot_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "scalability_summary.csv"
    slot_csv = output_dir / "scalability_slot_metrics.csv"

    summaries = []
    snapshot_metadata = None
    with slot_csv.open("w", newline="", encoding="utf-8") as stream:
        slot_writer = csv.DictWriter(stream, fieldnames=SLOT_FIELDS)
        slot_writer.writeheader()
        for sats_per_plane in args.sats_per_plane:
            scale_summaries, snapshot = run_scale(
                planes=args.planes,
                sats_per_plane=sats_per_plane,
                seed=args.seed,
                warmup_slots=args.warmup_slots,
                eval_slots=args.eval_slots,
                checkpoint=checkpoint,
                lydrl_checkpoint=lydrl_checkpoint,
                snapshot_dir=snapshot_dir,
                reuse_n192_snapshot=args.reuse_n192_snapshot,
                slot_writer=slot_writer,
            )
            summaries.extend(scale_summaries)
            if snapshot is not None:
                snapshot_metadata = snapshot

    summaries.sort(key=lambda row: (row["algorithm"], row["n_sats"]))
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    figure_paths: Iterable[Path] = []
    if not args.skip_plots:
        figure_paths = plot_results(summary_csv, output_dir / "figures")

    expected_rows = len(ALGORITHMS) * len(args.sats_per_plane) * args.eval_slots
    manifest = {
        "experiment": "five-policy zero-shot constellation scalability",
        "important_training_provenance": {
            "test_topologies": "16x{8,10,12,16,20}",
            "learned_policy_training_constellation": "5x5 (25 satellites)",
            "user_selected_resolution": "use 5x5-trained gold BLA-MAPPO on the requested topologies",
        },
        "algorithm_changed": False,
        "retraining_or_finetuning": False,
        "checkpoint": str(checkpoint.relative_to(HERE)),
        "actor_sha256": sha256_file(checkpoint / "actor.pth"),
        "lydrl_checkpoint": str(lydrl_checkpoint.relative_to(HERE)),
        "lydrl_actor_sha256": sha256_file(lydrl_checkpoint / "actor.pth"),
        "algorithms": list(ALGORITHMS),
        "seed": args.seed,
        "walker_sizes": [f"{args.planes}x{s}" for s in args.sats_per_plane],
        "n_satellites": [args.planes * s for s in args.sats_per_plane],
        "warmup_policy": "gold BLA-MAPPO",
        "warmup_slots_per_size": args.warmup_slots,
        "common_warmup_state_forked_across_algorithms": True,
        "eval_slots_per_size": args.eval_slots,
        "metric_definitions": {
            "user_satisfaction": "ontime / (completed + timeout)",
            "system_delay_overhead": "mean over slots of sum of completed-task E2E delays in that slot",
            "system_energy_overhead": "mean over slots of total compute plus transmission energy; housekeeping excluded",
            "total_lifetime_loss": "sum of per-satellite convex lifetime loss over all formal-evaluation slots",
        },
        "single_seed_no_confidence_interval": True,
        "slot_csv_rows": expected_rows,
        "snapshot_192": snapshot_metadata,
        "figures": [path.name for path in figure_paths],
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"summary CSV: {summary_csv}", flush=True)
    print(f"slot CSV: {slot_csv} ({expected_rows} rows)", flush=True)
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
