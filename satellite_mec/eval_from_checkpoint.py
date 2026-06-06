"""
eval_from_checkpoint.py
=======================
从已保存的检查点加载 MAPPO/MAPPO_NoDod 模型，
与所有基线一起评估，生成对比图表。

用法
----
    python eval_from_checkpoint.py \
        --mappo_path results/20260606_123236_MainComparison/MAPPO/model \
        --mappo_nodod_path results/20260606_123236_MainComparison/MAPPO_NoDod/checkpoints/step_57600 \
        --debug
"""

import argparse
import os
import sys

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
    p.add_argument('--mappo_path',       required=True,
                   help='MAPPO model directory')
    p.add_argument('--mappo_nodod_path', required=True,
                   help='MAPPO_NoDod checkpoint directory')
    p.add_argument('--debug',  action='store_true')
    p.add_argument('--n_runs', type=int, default=None)
    p.add_argument('--no_plots', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    runner = ExperimentRunner(cfg, 'EvalFromCheckpoint', debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)

    if args.n_runs is not None:
        cfg.N_EVAL_RUNS = args.n_runs

    # ── 建立目录 ──────────────────────────────────────────────
    for name in ['MAPPO', 'MAPPO_NoDod', 'LocalOnly',
                 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(name)

    # ── 实例化策略 ─────────────────────────────────────────────
    mappo        = MAPPOPolicy(cfg, name='MAPPO')
    mappo_no_dod = create_mappo_no_dod(cfg, env)
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)

    # ── 恢复 env 原始 lyapunov_calc（create_mappo_no_dod 会替换）──
    from core import LyapunovCalculator
    env.lyapunov_calc = LyapunovCalculator(cfg)

    # ── 加载已训练的模型 ───────────────────────────────────────
    mappo_path = os.path.abspath(args.mappo_path)
    logger.info(f"加载 MAPPO: {mappo_path}")
    mappo.load(mappo_path)

    nodod_path = os.path.abspath(args.mappo_nodod_path)
    logger.info(f"加载 MAPPO_NoDod: {nodod_path}")
    mappo_no_dod.load(nodod_path)

    policies = [mappo, mappo_no_dod, local_only, greedy_delay, lya_greedy, mhspo]

    # ── 预热（MHSPO 训练其 DOGD 预测器）─────────────────────────
    logger.info("预热阶段")
    runner.run_warmup(env, policy=mhspo)

    # ── 逐 run 评估 ───────────────────────────────────────────
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

    # ── 保存 & 绘图 ───────────────────────────────────────────
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
                           n_sats=cfg.N_SATS,
                           n_snapshots=n_snaps,
                           snapshot_interval=snapshot_interval)
        plot_delay_pdf(curves_by_run, fig_dir, 'delay_pdf', delay_sample_interval=5)
        plot_satisfaction_pdf(curves_by_run, fig_dir,
                              sample_key='satisfaction_samples_a',
                              filename='satisfaction_pdf')

    logger.info(f"评估完成！结果目录：{runner.base_dir}")

    # ── 打印摘要 ──────────────────────────────────────────────
    print("\n========== 评估结果汇总 ==========")
    for pol in policies:
        summary = runner.recorders[pol.name].get_summary()
        if summary:
            cr  = summary.get('completion_rate', {})
            hl  = summary.get('health_loss', {})
            sat = summary.get('slot_satisfaction_rate', {})
            print(f"  {pol.name:20s} CR={cr.get('mean', 0):.4f}±{cr.get('std', 0):.4f}  "
                  f"HL={hl.get('mean', 0):.4e}  Sat={sat.get('mean', 0):.4f}")


if __name__ == '__main__':
    main()
