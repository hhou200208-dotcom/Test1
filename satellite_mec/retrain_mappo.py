"""
retrain_mappo.py
================
重新训练 MAPPO（含完成奖励修复），然后与所有基线一起评估。

修复内容
--------
- 添加 COMPLETION_BONUS=1.0（任务完成奖励），直接激励CR
- 使用 MHSPO 预热（训练 DOGD 预测器）
- 跳过 L_MAX 校准（使用理论值）

用法
----
    python retrain_mappo.py [--debug] [--no_plots] [--t_train N]
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
from training import MAPPOPolicy, create_mappo_no_dod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug',    action='store_true')
    p.add_argument('--no_plots', action='store_true')
    p.add_argument('--t_train',  type=int, default=None)
    p.add_argument('--n_runs',   type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    if args.t_train is not None:
        cfg.T_TRAIN = args.t_train
    if args.n_runs is not None:
        cfg.N_EVAL_RUNS = args.n_runs

    runner = ExperimentRunner(cfg, 'MAPPO_WithBonus', debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)
    logger.info(f"COMPLETION_BONUS = {cfg.COMPLETION_BONUS}")

    for name in ['MAPPO', 'MAPPO_NoDod', 'LocalOnly',
                 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(name)

    mappo        = MAPPOPolicy(cfg, name='MAPPO')
    mappo_no_dod = create_mappo_no_dod(cfg, env)
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)

    env.lyapunov_calc = LyapunovCalculator(cfg)

    # ── 训练 ──────────────────────────────────────────────────
    logger.info("训练 MAPPO")
    runner.run_training(mappo, env)
    logger.info("训练 MAPPO_NoDod")
    runner.run_training(mappo_no_dod, env)

    # ── 预热 ──────────────────────────────────────────────────
    logger.info("预热（MHSPO DOGD）")
    runner.run_warmup(env, policy=mhspo)

    # ── 评估 ──────────────────────────────────────────────────
    policies = [mappo, mappo_no_dod, local_only, greedy_delay, lya_greedy, mhspo]
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

    if not args.no_plots:
        fig_dir = os.path.join(runner.base_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        METRICS = [
            ('completion_rate',        'Task Completion Rate',    'Completion Rate'),
            ('avg_dod',                'Average DoD',              'Average DoD'),
            ('health_loss',            'Cumulative Health Loss',   'Cumulative Health Loss'),
            ('total_queue',            'Total Queue Backlog (MB)', 'Total Queue (QF+QB)'),
            ('forwarded',              'Forwarded Tasks / Slot',   'Forwarded Tasks'),
            ('slot_satisfaction_rate', 'User Satisfaction Rate',
             'User Satisfaction Rate (satisfied/(done+timeout))'),
        ]
        for metric, ylabel, title in METRICS:
            plot_metric_per_run(curves_by_run, metric, ylabel, title, fig_dir, metric)

        n_snaps = cfg.T_EVAL // snapshot_interval
        plot_dod_snapshots(snapshots_by_run,
                           output_dir=os.path.join(fig_dir, 'dod_snapshots'),
                           n_sats=cfg.N_SATS, n_snapshots=n_snaps,
                           snapshot_interval=snapshot_interval)
        plot_delay_pdf(curves_by_run, fig_dir, 'delay_pdf', delay_sample_interval=5)
        plot_satisfaction_pdf(curves_by_run, fig_dir,
                              sample_key='satisfaction_samples_a',
                              filename='satisfaction_pdf')

    # ── 摘要 ──────────────────────────────────────────────────
    print("\n========== 评估结果汇总 ==========")
    for pol in policies:
        summary = runner.recorders[pol.name].get_summary()
        if summary:
            cr  = summary.get('completion_rate', {})
            hl  = summary.get('health_loss', {})
            sat = summary.get('slot_satisfaction_rate', {})
            print(f"  {pol.name:20s} CR={cr.get('mean', 0):.4f}±{cr.get('std', 0):.4f}  "
                  f"HL={hl.get('mean', 0):.4e}  Sat={sat.get('mean', 0):.4f}")

    logger.info(f"全部完成！结果目录：{runner.base_dir}")


if __name__ == '__main__':
    main()
