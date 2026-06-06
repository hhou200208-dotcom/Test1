"""
sensitivity_v.py
================
V 值敏感性分析（LAMBDA_HIGH=2.0）。

对 V ∈ {10, 50, 100, 200, 500} 分别训练 MAPPO 并评估全部 6 个策略，
最后绘制 CR/DoD/HL/Sat vs V 的折线图。

用法
----
    python sensitivity_v.py [--debug] [--no_plots] [--t_train N]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import SensitivityVConfig, SatelliteMECEnv, LyapunovCalculator
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)
from evaluation import MetricsRecorder, ExperimentRunner
from training import MAPPOPolicy

V_GRID = [10.0, 50.0, 100.0, 200.0, 500.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--debug',    action='store_true')
    p.add_argument('--no_plots', action='store_true')
    p.add_argument('--t_train',  type=int, default=None)
    p.add_argument('--n_runs',   type=int, default=None)
    p.add_argument('--v_grid',   type=str, default=None,
                   help='Comma-separated V values, e.g. "10,50,100"')
    return p.parse_args()


def run_one_v(v_value: float, args, parent_dir: str):
    """对单个 V 值执行训练 + 评估，返回 summary dict。"""
    cfg = SensitivityVConfig(v_value)
    if args.t_train is not None:
        cfg.T_TRAIN = args.t_train
    if args.n_runs is not None:
        cfg.N_EVAL_RUNS = args.n_runs

    tag = f'V{int(v_value)}'
    runner = ExperimentRunner(cfg, f'SensV_{tag}', debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)
    logger.info(f"=== V = {v_value} | LAMBDA_HIGH = {cfg.LAMBDA_HIGH} ===")
    logger.info(f"COMPLETION_BONUS = {cfg.COMPLETION_BONUS}")

    for name in ['MAPPO', 'LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(name)

    lyapunov_default = LyapunovCalculator(cfg)
    mappo        = MAPPOPolicy(cfg, name='MAPPO')
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)

    # 训练
    logger.info("训练 MAPPO")
    env.lyapunov_calc = lyapunov_default
    runner.run_training(mappo, env)

    # 预热
    logger.info("预热（MHSPO DOGD）")
    runner.run_warmup(env, policy=mhspo)

    # 评估
    policies = [mappo, local_only, greedy_delay, lya_greedy, mhspo]
    logger.info(f"评估：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")

    for run_idx in range(cfg.N_EVAL_RUNS):
        seeds = cfg.get_eval_seeds(run_idx)
        run_recorders = {pol.name: MetricsRecorder(cfg, pol.name) for pol in policies}

        for policy in policies:
            env.reset(phase='eval', seeds=seeds)
            policy.set_eval_mode()
            rec = run_recorders[policy.name]
            iterator = (tqdm(range(cfg.T_EVAL),
                             desc=f'[V{int(v_value)}|{policy.name}] run{run_idx}',
                             ncols=100, unit='slot')
                        if HAS_TQDM else range(cfg.T_EVAL))
            for _ in iterator:
                _, _, _, info = env.step(policy=policy)
                rec.record_slot(info, phase='eval')
            rec.record_eval_run(run_idx)
            cr = env.get_eval_completion_rate()
            logger.info(f"  [{policy.name}] run{run_idx}: CR={cr:.4f}")

        for pol in policies:
            runner.recorders[pol.name]._slot_records.extend(
                run_recorders[pol.name]._slot_records)
            runner.recorders[pol.name]._eval_run_records.extend(
                run_recorders[pol.name]._eval_run_records)

    runner.save_all_results(generate_plots=False)

    # 收集 summary
    summary = {'V': v_value, 'lambda_high': cfg.LAMBDA_HIGH}
    for pol in policies:
        s = runner.recorders[pol.name].get_summary()
        if s:
            summary[pol.name] = {
                'CR':  s.get('completion_rate', {}).get('mean', 0),
                'CR_std': s.get('completion_rate', {}).get('std', 0),
                'DoD': s.get('avg_dod', {}).get('mean', 0),
                'HL':  s.get('avg_health_loss', {}).get('mean', 0),
                'Z':   s.get('avg_z', {}).get('mean', 0),
            }
    logger.info(f"=== V={v_value} 完成 ===")
    return summary, runner.base_dir


def main():
    args = parse_args()
    if args.v_grid:
        v_grid = [float(v) for v in args.v_grid.split(',')]
    else:
        v_grid = V_GRID

    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'results', time.strftime('%Y%m%d_%H%M%S') + '_SensitivityV')
    os.makedirs(parent_dir, exist_ok=True)
    print(f"敏感性分析父目录：{parent_dir}")

    all_results = []
    for v in v_grid:
        print(f"\n{'='*70}\n  V = {v}\n{'='*70}\n")
        t0 = time.time()
        summary, sub_dir = run_one_v(v, args, parent_dir)
        all_results.append(summary)
        print(f"\n[V={v}] 完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
        # 中间保存
        with open(os.path.join(parent_dir, 'sensitivity_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # 最终打印
    print("\n" + "="*100)
    print("V 敏感性分析汇总（LAMBDA_HIGH=2.0）")
    print("="*100)
    print(f"{'V':>8s} {'MAPPO_CR':>12s} {'MAPPO_DoD':>12s} {'MAPPO_HL':>15s} "
          f"{'MHSPO_CR':>12s} {'MHSPO_DoD':>12s}")
    for r in all_results:
        v = r['V']
        m = r.get('MAPPO', {})
        h = r.get('MHSPO', {})
        print(f"{v:>8.0f} {m.get('CR',0):>12.4f} {m.get('DoD',0):>12.4f} "
              f"{m.get('HL',0):>15.3e} {h.get('CR',0):>12.4f} {h.get('DoD',0):>12.4f}")

    # 绘图
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            vs = [r['V'] for r in all_results]
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            metrics = [('CR', 'Completion Rate', axes[0, 0]),
                       ('DoD', 'Average DoD', axes[0, 1]),
                       ('HL', 'Avg Health Loss / slot', axes[1, 0]),
                       ('Z', 'Avg Virtual Queue Z', axes[1, 1])]
            policies = ['MAPPO', 'MHSPO', 'LyapunovGreedy', 'GreedyDelay', 'LocalOnly']
            for key, ylabel, ax in metrics:
                for pol in policies:
                    ys = [r.get(pol, {}).get(key, None) for r in all_results]
                    if all(y is not None for y in ys):
                        ax.plot(vs, ys, marker='o', label=pol, linewidth=2)
                ax.set_xlabel('V')
                ax.set_ylabel(ylabel)
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_title(f'{ylabel} vs V')
            fig.suptitle(f'V Sensitivity (LAMBDA_HIGH=2.0)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(parent_dir, 'sensitivity_v.png'),
                        dpi=120, bbox_inches='tight')
            plt.close()
            print(f"\n图表已保存: {parent_dir}/sensitivity_v.png")
        except Exception as e:
            print(f"[警告] 绘图失败：{e}")

    print(f"\n全部完成！结果目录：{parent_dir}")


if __name__ == '__main__':
    main()
