"""
lambda_sweep.py
===============
扫描 LAMBDA_HIGH ∈ [2.0, 2.5]，每个值跑 1 轮基线评估，
汇总 CR / DoD / HL / Sat / Queue，绘制各指标 vs LAMBDA_HIGH 曲线。

用法
----
    python lambda_sweep.py [--lambda_grid 2.0,2.1,...]
"""

import argparse
import json
import os
import time
from statistics import mean

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


LAMBDA_GRID = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]


def make_config(lh: float) -> Config:
    """构造 LAMBDA_HIGH=lh 的 Config（n_runs=1 加速）。"""
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
        N_EVAL_RUNS = 1
    return _C()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda_grid', type=str, default=None,
                   help='Comma-separated LAMBDA_HIGH values')
    return p.parse_args()


def run_one_lambda(lh: float):
    cfg = make_config(lh)
    runner = ExperimentRunner(cfg, f'LambdaSweep_{lh:.1f}')
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)
    logger.info(f"=== LAMBDA_HIGH = {lh} | LAMBDA = {cfg.LAMBDA:.3f} ===")

    for name in ['LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        runner.setup_algorithm_dir(name)

    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)
    policies = [local_only, greedy_delay, lya_greedy, mhspo]

    logger.info("预热（MHSPO DOGD）")
    runner.run_warmup(env, policy=mhspo)

    logger.info(f"评估：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")
    for run_idx in range(cfg.N_EVAL_RUNS):
        seeds = cfg.get_eval_seeds(run_idx)
        for policy in policies:
            env.reset(phase='eval', seeds=seeds)
            policy.set_eval_mode()
            rec = runner.recorders[policy.name]
            iterator = (tqdm(range(cfg.T_EVAL),
                             desc=f'[lh={lh:.1f}|{policy.name}]',
                             ncols=100, unit='slot')
                        if HAS_TQDM else range(cfg.T_EVAL))
            for _ in iterator:
                _, _, _, info = env.step(policy=policy)
                rec.record_slot(info, phase='eval')
            rec.record_eval_run(run_idx)
            cr = env.get_eval_completion_rate()
            logger.info(f"  [{policy.name}]: CR={cr:.4f}")

    runner.save_all_results(generate_plots=False)

    # 收集指标
    res = {'lambda_high': lh, 'lambda': cfg.LAMBDA}
    for pol in policies:
        with open(os.path.join(runner.base_dir, pol.name, 'eval_runs.json')) as f:
            r = json.load(f)[0]
        with open(os.path.join(runner.base_dir, pol.name, 'curves.json')) as f:
            c = json.load(f)
        sat_curve = c['satisfaction_curve']['slot_satisfaction_rate']
        qf = c['queue_curve']['avg_qf_size']
        qb = c['queue_curve']['avg_qb_size']
        q_mb = [(a+b)/1e6 for a,b in zip(qf, qb)]
        res[pol.name] = {
            'CR':  r['completion_rate'],
            'DoD': r['avg_dod'],
            'HL':  r['avg_health_loss'],
            'Sat': mean(sat_curve),
            'Q':   mean(q_mb),
        }
    return res, runner.base_dir


def main():
    args = parse_args()
    grid = ([float(x) for x in args.lambda_grid.split(',')]
            if args.lambda_grid else LAMBDA_GRID)

    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'results', time.strftime('%Y%m%d_%H%M%S') + '_LambdaSweep')
    os.makedirs(parent_dir, exist_ok=True)
    print(f"扫描父目录：{parent_dir}")

    all_results = []
    for lh in grid:
        print(f"\n{'='*70}\n  LAMBDA_HIGH = {lh}\n{'='*70}\n")
        t0 = time.time()
        res, sub_dir = run_one_lambda(lh)
        all_results.append(res)
        print(f"\n[lh={lh}] 完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
        with open(os.path.join(parent_dir, 'lambda_sweep_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # 汇总打印
    print("\n" + "="*120)
    print(f"{'λ_h':>6s} {'λ':>6s} | " +
          " | ".join(f"{p}_CR".rjust(10) for p in ['Loc','Gre','Lya','MHS']) +
          " | " + " | ".join(f"{p}_DoD".rjust(10) for p in ['Loc','Gre','Lya','MHS']))
    for r in all_results:
        crs = [r.get(p, {}).get('CR', 0) for p in ['LocalOnly','GreedyDelay','LyapunovGreedy','MHSPO']]
        dods = [r.get(p, {}).get('DoD', 0) for p in ['LocalOnly','GreedyDelay','LyapunovGreedy','MHSPO']]
        print(f"{r['lambda_high']:>6.2f} {r['lambda']:>6.3f} | " +
              " | ".join(f"{c:>10.4f}" for c in crs) +
              " | " + " | ".join(f"{d:>10.4f}" for d in dods))

    # 绘图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        lhs = [r['lambda_high'] for r in all_results]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        policies = ['LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']
        colors = {'LocalOnly': 'C0', 'GreedyDelay': 'C1', 'LyapunovGreedy': 'C2', 'MHSPO': 'C3'}
        for key, ylabel, ax in [('CR', 'Completion Rate', axes[0,0]),
                                 ('DoD', 'Average DoD', axes[0,1]),
                                 ('HL', 'Avg Health Loss / slot', axes[1,0]),
                                 ('Sat', 'Satisfaction Rate (mean)', axes[1,1])]:
            for pol in policies:
                ys = [r.get(pol, {}).get(key, None) for r in all_results]
                ax.plot(lhs, ys, marker='o', label=pol, color=colors[pol], linewidth=2)
            ax.set_xlabel('LAMBDA_HIGH')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_title(f'{ylabel} vs LAMBDA_HIGH')
        fig.suptitle('Lambda Sweep (n_runs=1)', fontsize=14)
        plt.tight_layout()
        fig_path = os.path.join(parent_dir, 'lambda_sweep.png')
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\n图表已保存：{fig_path}")
    except Exception as e:
        print(f"[警告] 绘图失败：{e}")

    print(f"\n全部完成！结果目录：{parent_dir}")


if __name__ == '__main__':
    main()
