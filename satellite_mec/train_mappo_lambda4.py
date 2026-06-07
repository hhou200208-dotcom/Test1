"""train_mappo_lambda4.py
=========================
Train MAPPO at λ_high=4 (main paper scenario), then evaluate alongside
the 4 baselines.

Codex Stages 1-4 are active in core/env.py + core/satellite.py +
core/config.py + training/policy.py (commit 773c159):
  - sequential decision path
  - outcome-aware reward
  - 54-dim Actor state, 235-dim Critic state
  - eval_timeout bug fix

Usage
-----
    python train_mappo_lambda4.py [--t_train N] [--n_runs M] [--no_plots]
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv, LyapunovCalculator
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)
from evaluation import MetricsRecorder, ExperimentRunner
from evaluation.plotting import (
    plot_metric_per_run, plot_dod_snapshots,
    plot_delay_pdf, plot_satisfaction_pdf,
    _collect_baseline_curves, _get_dod_snapshots,
)
from training import MAPPOPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda_high', type=float, default=4.0)
    p.add_argument('--t_train',     type=int,   default=None)
    p.add_argument('--n_runs',      type=int,   default=1)
    p.add_argument('--no_plots',    action='store_true')
    p.add_argument('--debug',       action='store_true')
    return p.parse_args()


def make_config(lh: float, args) -> Config:
    """Build Config with overridden LAMBDA_HIGH."""
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA      = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    cfg = _C()
    if args.t_train is not None:
        cfg.T_TRAIN = args.t_train
    if args.n_runs is not None:
        cfg.N_EVAL_RUNS = args.n_runs
    return cfg


def main():
    args = parse_args()
    cfg  = make_config(args.lambda_high, args)

    runner = ExperimentRunner(cfg, f'MAPPO_lh{args.lambda_high:.1f}',
                              debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)
    logger.info(f"=== λ_high={args.lambda_high}, λ={cfg.LAMBDA:.3f}, "
                f"T_TRAIN={cfg.T_TRAIN}, n_runs={cfg.N_EVAL_RUNS} ===")
    logger.info(f"State dim {cfg.get_state_dim()}, Critic dim {cfg.get_critic_state_dim()}")

    for name in ['MAPPO', 'LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(name)

    lyapunov_default = LyapunovCalculator(cfg)

    mappo        = MAPPOPolicy(cfg, name='MAPPO')
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)

    # ── 训练 MAPPO ────────────────────────────────────────────
    logger.info("训练 MAPPO（sequential + outcome-aware reward）")
    env.lyapunov_calc = lyapunov_default
    runner.run_training(mappo, env)

    # ── 预热 MHSPO DOGD 预测器 ────────────────────────────────
    logger.info("预热 MHSPO DOGD")
    runner.run_warmup(env, policy=mhspo)

    # ── 评估 ──────────────────────────────────────────────────
    policies = [mappo, local_only, greedy_delay, lya_greedy, mhspo]
    curves_by_run    = []
    snapshots_by_run = []
    snapshot_interval = 360

    logger.info(f"评估：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")

    for run_idx in range(cfg.N_EVAL_RUNS):
        seeds = cfg.get_eval_seeds(run_idx)
        run_recorders = {pol.name: MetricsRecorder(cfg, pol.name) for pol in policies}

        for policy in policies:
            env.reset(phase='eval', seeds=seeds)
            policy.set_eval_mode()
            rec = run_recorders[policy.name]
            iterator = (tqdm(range(cfg.T_EVAL),
                             desc=f'[{policy.name}] run{run_idx}',
                             ncols=100, unit='slot')
                        if HAS_TQDM else range(cfg.T_EVAL))
            for _ in iterator:
                _, _, _, info = env.step(policy=policy)
                rec.record_slot(info, phase='eval')
            rec.record_eval_run(run_idx)
            cr = env.get_eval_completion_rate()
            logger.info(f"  [{policy.name}] run{run_idx}: CR={cr:.4f}")

        curves_by_run.append(_collect_baseline_curves(run_recorders))
        snapshots_by_run.append(_get_dod_snapshots(
            run_recorders, snapshot_interval=snapshot_interval))

        for pol in policies:
            runner.recorders[pol.name]._slot_records.extend(
                run_recorders[pol.name]._slot_records)
            runner.recorders[pol.name]._eval_run_records.extend(
                run_recorders[pol.name]._eval_run_records)

    runner.save_all_results(generate_plots=False)

    # ── 摘要：5 项标准指标 ────────────────────────────────────
    print("\n" + "=" * 90)
    print("  评估结果汇总 (λ_high={:.1f}) — 5 项标准指标".format(args.lambda_high))
    print("=" * 90)
    print(f"  {'策略':<16}{'CR':>8}{'满意度':>9}{'时延(s)':>10}"
          f"{'HL/slot':>14}{'DoD':>8}{'队列MB':>9}")
    print("-" * 90)
    for pol in policies:
        summary = runner.recorders[pol.name].get_summary()
        if not summary:
            continue
        cr  = summary.get('completion_rate', {}).get('mean', 0)
        sat = summary.get('avg_satisfaction_rate', {}).get('mean', 0)
        dly = summary.get('avg_e2e_delay', {}).get('mean', 0)
        hl  = summary.get('avg_health_loss', {}).get('mean', 0)
        dod = summary.get('avg_dod', {}).get('mean', 0)
        qmb = summary.get('avg_queue_mb', {}).get('mean', 0)
        print(f"  {pol.name:<14}{cr:>8.4f}{sat:>9.4f}{dly:>10.3f}"
              f"{hl:>14.3e}{dod:>8.4f}{qmb:>9.2f}")

    logger.info(f"全部完成！结果目录：{runner.base_dir}")


if __name__ == '__main__':
    main()
