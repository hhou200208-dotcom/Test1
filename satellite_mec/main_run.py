"""
main_run.py
===========
实验入口（唯一入口，职责：参数解析 + 编排实验流程）。

实验模式
--------
    python main_run.py                    # 完整基线评估（N_EVAL_RUNS=5）
    python main_run.py --debug            # 调试模式（快速验证流程是否通）
    python main_run.py --mode comparison  # 含MAPPO训练的完整对比实验
    python main_run.py --mode ablation    # 消融实验
    python main_run.py --mode sensitivity # V值敏感性分析

关键修复（相比原始代码）
------------------------
[BUG-FIX] run_baselines_only() 中 `range(1)` → `range(cfg.N_EVAL_RUNS)`
    原始 bug：只跑1次评估导致 baseline 统计不可靠
    修复：恢复为 N_EVAL_RUNS=5 次独立种子评估

[WARMUP-FIX] 确保评估前预热与评估独立
    原始问题：预热和评估共用同一段时隙，导致评估期落在暂态
    修复：run_warmup() 独立运行，评估时 reset 到指定种子
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv, LyapunovCalculator
from core.config import AblationConfig, SensitivityVConfig
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)
from evaluation import MetricsRecorder, ExperimentRunner
from evaluation.plotting import (
    plot_metric_per_run, plot_dod_snapshots,
    plot_delay_pdf, plot_satisfaction_pdf,
    _collect_baseline_curves, _get_dod_snapshots,
)


# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='卫星MEC调度策略对比实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', choices=['baselines', 'comparison', 'ablation', 'sensitivity'],
                        default='baselines', help='实验模式')
    parser.add_argument('--debug',    action='store_true', help='调试模式（快速运行）')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--t_train',  type=int, default=None)
    parser.add_argument('--t_eval',   type=int, default=None)
    parser.add_argument('--n_runs',   type=int, default=None,
                        help='评估run次数（覆盖配置中的 N_EVAL_RUNS）')
    parser.add_argument('--no_plots', action='store_true')
    return parser.parse_args()


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed != 42:
        cfg.SEED = args.seed; cfg.SEED_TASK = args.seed
        cfg.SEED_TASK_PARAM = args.seed + 1
        cfg.SEED_NET = args.seed + 3; cfg.SEED_TRAIN = args.seed + 4
    if args.t_train is not None: cfg.T_TRAIN = args.t_train
    if args.t_eval  is not None: cfg.T_EVAL  = args.t_eval
    if args.n_runs  is not None: cfg.N_EVAL_RUNS = args.n_runs
    return cfg


# ──────────────────────────────────────────────────────────────
# 核心实验：仅基线（不含RL训练）
# ──────────────────────────────────────────────────────────────
def run_baselines_only(args: argparse.Namespace) -> None:
    """
    评估4个确定性基线策略。

    [BUG-FIX] 原始代码 `range(1)` 改为 `range(cfg.N_EVAL_RUNS)`，
    保证 5 次独立种子评估，统计数据才具有代表性。
    """
    cfg = apply_args(Config(), args)
    runner = ExperimentRunner(cfg, 'BaselinesOnly', debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)

    for pol_name in ['LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(pol_name)

    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)
    policies = [local_only, greedy_delay, lya_greedy, mhspo]

    # ── 预热（隔离于评估，避免暂态影响）──────────────────────────
    # 用 MHSPO 预热以训练其 DOGD 预测器：env.reset 会清队列，
    # 故对其他基线无影响；MHSPO 的预测器作为 policy 状态保留。
    logger.info("预热阶段")
    runner.run_warmup(env, policy=mhspo)

    # ── 逐 run 评估 ───────────────────────────────────────────
    curves_by_run:    List[Dict] = []
    snapshots_by_run: List[Dict] = []
    snapshot_interval = 360

    logger.info(f"评估阶段：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")

    # ╔════════════════════════════════════════════════════════╗
    # ║  [BUG-FIX] 原始代码：range(1) → 只跑1次，统计不可靠   ║
    # ║  修复后：range(cfg.N_EVAL_RUNS) = range(5)            ║
    # ╚════════════════════════════════════════════════════════╝
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
            policy.set_train_mode()
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

    logger.info(f"基线评估完成！结果目录：{runner.base_dir}")


# ──────────────────────────────────────────────────────────────
# 含 MAPPO 的完整对比实验
# ──────────────────────────────────────────────────────────────
def run_full_comparison(args: argparse.Namespace) -> None:
    """含 MAPPO 训练的完整对比实验（耗时数小时）。"""
    from training import MAPPOPolicy, create_mappo_no_dod

    cfg    = apply_args(Config(), args)
    runner = ExperimentRunner(cfg, 'MainComparison', debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)

    runner.setup_algorithm_dir('MAPPO')
    runner.setup_algorithm_dir('MAPPO_NoDod')
    for name in ['LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(name)

    mappo        = MAPPOPolicy(cfg, name='MAPPO')
    mappo_no_dod = create_mappo_no_dod(cfg, env)
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env)

    logger.info("L_MAX 校准阶段")
    runner.calibrate_l_max(env)
    logger.info("训练阶段")
    runner.run_training(mappo, env)
    runner.run_training(mappo_no_dod, env)
    logger.info("预热阶段")
    runner.run_warmup(env)
    logger.info("评估阶段")
    runner.run_evaluation([mappo, mappo_no_dod, local_only, greedy_delay, lya_greedy, mhspo], env)
    runner.save_all_results(generate_plots=not args.no_plots)
    logger.info("完整对比实验完成！")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'baselines':
        run_baselines_only(args)
    elif args.mode == 'comparison':
        run_full_comparison(args)
    else:
        print(f"[main_run] mode={args.mode} 暂未在此文件实现，请参考原始代码。")
