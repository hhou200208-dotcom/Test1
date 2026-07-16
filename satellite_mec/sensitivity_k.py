"""sensitivity_k.py
==================
多跳转发次数 K（K_MAX）敏感性分析。

方法学说明
----------
本实验采用 **test-time（评估期）敏感性分析**：固定加载论文头号模型
``checkpoints/LyaMAPPO_lh4_32K``（该模型在 λ_high=4.0、K_MAX=3 场景下训练，
且为不可复现的 golden checkpoint，见 checkpoints/README.md），
在评估阶段将最大转发跳数 ``cfg.K_MAX`` 依次设为 K ∈ {1,2,3,4,5}，
其余场景参数（λ_high=4.0、V=50、seeds 等）与论文主实验完全一致。
由此考察 **同一策略对多跳转发预算 K 的鲁棒性**。

纵坐标
------
    1. 用户满意度（口径A）: avg_satisfaction_rate
       = 每时隙 slot_satisfaction_rate 的均值，口径 satisfied/(done+timeout)
    2. 累计电池寿命损耗: cumulative_health_loss
       = 评估期内每时隙 avg_health_loss 之和（对应论文式(21) L 的累计）

注意
----
K=3 为模型的训练取值，应最接近论文头号结果（可作对齐 sanity check）；
K<3 / K>3 均为策略未见过的转发预算，用于观察敏感性。

用法
----
    python sensitivity_k.py                       # 全量：K∈{1,2,3,4,5}, n_runs=5
    python sensitivity_k.py --debug               # 快速冒烟（短评估、1 run）
    python sensitivity_k.py --k_grid 1,3,5 --n_runs 3
"""

import argparse
import json
import os
import time

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv, LyapunovCalculator
from evaluation import MetricsRecorder, ExperimentRunner
from training import MAPPOPolicy

K_GRID_DEFAULT = [1, 2, 3, 4, 5]
CKPT_DEFAULT = 'checkpoints/LyaMAPPO_lh4_32K'


def parse_args():
    p = argparse.ArgumentParser(description='K_MAX（多跳转发次数）敏感性分析')
    p.add_argument('--ckpt',        type=str,   default=CKPT_DEFAULT,
                   help='已训练 MAPPO 模型目录（默认 golden LyaMAPPO_lh4_32K）')
    p.add_argument('--lambda_high', type=float, default=4.0,
                   help='高负载到达率（必须与训练场景一致，默认 4.0）')
    p.add_argument('--k_grid',      type=str,   default=None,
                   help='逗号分隔的 K 取值，如 "1,2,3,4,5"')
    p.add_argument('--n_runs',      type=int,   default=None,
                   help='每个 K 的独立评估 run 数（覆盖 N_EVAL_RUNS）')
    p.add_argument('--t_eval',      type=int,   default=None)
    p.add_argument('--debug',       action='store_true')
    p.add_argument('--no_plots',    action='store_true')
    return p.parse_args()


def make_config(lh: float, k: int, args) -> Config:
    """构造 λ_high=lh、K_MAX=k 的评估配置（其余同论文主实验）。"""
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA      = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    cfg = _C()
    cfg.K_MAX = int(k)                 # K_MAX 在 satellite.py 中实时读取，评估期覆盖即可
    if args.t_eval is not None:
        cfg.T_EVAL = args.t_eval
    if args.n_runs is not None:
        cfg.N_EVAL_RUNS = args.n_runs
    return cfg


def eval_one_k(k: int, args, ckpt_abs: str):
    """加载 golden 模型，在 K_MAX=k 下评估 N_EVAL_RUNS 次，返回 summary。"""
    cfg = make_config(args.lambda_high, k, args)
    if args.debug:
        cfg.T_EVAL = cfg.DEBUG_T_EVAL
        cfg.N_EVAL_RUNS = cfg.DEBUG_N_EVAL_RUNS

    runner = ExperimentRunner(cfg, f'SensK_K{k}', debug=args.debug)
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)
    env.lyapunov_calc = LyapunovCalculator(cfg)
    runner.setup_algorithm_dir('MAPPO')

    logger.info(f"=== K_MAX={k} | λ_high={cfg.LAMBDA_HIGH} λ={cfg.LAMBDA:.3f} "
                f"| state_dim={cfg.get_state_dim()} critic_dim={cfg.get_critic_state_dim()} ===")

    # 固定加载 golden 模型（每个 K 重新实例化，避免任何状态泄漏）
    mappo = MAPPOPolicy(cfg, lyapunov_calc=LyapunovCalculator(cfg), name='MAPPO')
    mappo.load(ckpt_abs)

    logger.info(f"评估：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")
    for run_idx in range(cfg.N_EVAL_RUNS):
        seeds = cfg.get_eval_seeds(run_idx)
        env.reset(phase='eval', seeds=seeds)
        mappo.set_eval_mode()
        rec = runner.recorders['MAPPO']
        iterator = (tqdm(range(cfg.T_EVAL), desc=f'[K={k}] run{run_idx}',
                         ncols=100, unit='slot') if HAS_TQDM else range(cfg.T_EVAL))
        for _ in iterator:
            _, _, _, info = env.step(policy=mappo)
            rec.record_slot(info, phase='eval')
        rec.record_eval_run(run_idx)
        cr = env.get_eval_completion_rate()
        logger.info(f"  [MAPPO] K={k} run{run_idx}: CR={cr:.4f}")

    runner.save_all_results(generate_plots=False)

    s = runner.recorders['MAPPO'].get_summary()
    sat = s.get('avg_satisfaction_rate', {})
    hlc = s.get('cumulative_health_loss', {})
    out = {
        'K': k,
        'lambda_high': cfg.LAMBDA_HIGH,
        'n_runs': s.get('n_runs', 0),
        't_eval': cfg.T_EVAL,
        # 主纵坐标 1：用户满意度（口径A）
        'satisfaction_mean': sat.get('mean', 0.0),
        'satisfaction_ci95': sat.get('ci95', 0.0),
        'satisfaction_std':  sat.get('std', 0.0),
        # 主纵坐标 2：累计电池寿命损耗
        'cum_health_loss_mean': hlc.get('mean', 0.0),
        'cum_health_loss_ci95': hlc.get('ci95', 0.0),
        'cum_health_loss_std':  hlc.get('std', 0.0),
        # 辅助上下文指标
        'completion_rate_mean': s.get('completion_rate', {}).get('mean', 0.0),
        'avg_health_loss_mean': s.get('avg_health_loss', {}).get('mean', 0.0),
        'avg_dod_mean':         s.get('avg_dod', {}).get('mean', 0.0),
        'avg_e2e_delay_mean':   s.get('avg_e2e_delay', {}).get('mean', 0.0),
        'avg_queue_mb_mean':    s.get('avg_queue_mb', {}).get('mean', 0.0),
    }
    logger.info(f"=== K={k} 完成: Sat={out['satisfaction_mean']:.4f} "
                f"CumHL={out['cum_health_loss_mean']:.4e} ===")
    return out


def plot_results(all_results, parent_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_results = sorted(all_results, key=lambda r: r['K'])
    ks   = [r['K'] for r in all_results]
    sat  = [r['satisfaction_mean'] for r in all_results]
    sat_e = [r['satisfaction_ci95'] for r in all_results]
    hl   = [r['cum_health_loss_mean'] for r in all_results]
    hl_e = [r['cum_health_loss_ci95'] for r in all_results]

    C_SAT, C_HL = '#1f77b4', '#d62728'

    # 图1：用户满意度 vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(ks, sat, yerr=sat_e, marker='o', color=C_SAT, linewidth=2,
                capsize=4, markersize=7)
    ax.set_xlabel('Max forwarding hops $K$')
    ax.set_ylabel('User Satisfaction Rate')
    ax.set_title('User Satisfaction vs. Max Forwarding Hops $K$')
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(parent_dir, 'k_satisfaction.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图2：累计电池寿命损耗 vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(ks, hl, yerr=hl_e, marker='s', color=C_HL, linewidth=2,
                capsize=4, markersize=7)
    ax.set_xlabel('Max forwarding hops $K$')
    ax.set_ylabel('Cumulative Battery Health Loss')
    ax.set_title('Battery Life Degradation vs. Max Forwarding Hops $K$')
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(parent_dir, 'k_health_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 图3：双轴联合面板
    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ln1 = ax1.errorbar(ks, sat, yerr=sat_e, marker='o', color=C_SAT, linewidth=2,
                       capsize=4, markersize=7, label='User Satisfaction')
    ax1.set_xlabel('Max forwarding hops $K$')
    ax1.set_ylabel('User Satisfaction Rate', color=C_SAT)
    ax1.tick_params(axis='y', labelcolor=C_SAT)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ln2 = ax2.errorbar(ks, hl, yerr=hl_e, marker='s', color=C_HL, linewidth=2,
                       capsize=4, markersize=7, linestyle='--', label='Cumulative Health Loss')
    ax2.set_ylabel('Cumulative Battery Health Loss', color=C_HL)
    ax2.tick_params(axis='y', labelcolor=C_HL)

    ax1.legend([ln1, ln2], ['User Satisfaction', 'Cumulative Health Loss'],
               loc='best', fontsize=9)
    ax1.set_title('Sensitivity to Max Forwarding Hops $K$'
                  f'  ($\\lambda_{{high}}$={all_results[0]["lambda_high"]}, LyaMAPPO)')
    fig.tight_layout()
    fig.savefig(os.path.join(parent_dir, 'k_sensitivity_panel.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"图表已保存: {parent_dir}/k_satisfaction.png, k_health_loss.png, k_sensitivity_panel.png")


def main():
    args = parse_args()
    ckpt_abs = os.path.abspath(args.ckpt)
    if not os.path.isdir(ckpt_abs):
        raise SystemExit(f"[错误] 找不到模型目录: {ckpt_abs}")

    if args.k_grid:
        k_grid = [int(x) for x in args.k_grid.split(',')]
    else:
        k_grid = K_GRID_DEFAULT

    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'results', time.strftime('%Y%m%d_%H%M%S') + '_SensitivityK')
    os.makedirs(parent_dir, exist_ok=True)
    print(f"K 敏感性分析父目录：{parent_dir}")
    print(f"加载模型：{ckpt_abs}")
    print(f"K 取值：{k_grid} | λ_high={args.lambda_high}")

    all_results = []
    for k in k_grid:
        print(f"\n{'='*70}\n  K_MAX = {k}\n{'='*70}")
        t0 = time.time()
        res = eval_one_k(k, args, ckpt_abs)
        all_results.append(res)
        print(f"[K={k}] 完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
        with open(os.path.join(parent_dir, 'sensitivity_k_results.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 汇总表
    print("\n" + "=" * 78)
    print(f"  K 敏感性分析汇总（LyaMAPPO_lh4_32K, λ_high={args.lambda_high}）")
    print("=" * 78)
    print(f"{'K':>3} {'满意度(口径A)':>16} {'累计寿命损耗':>18} {'CR':>8} {'HL/slot':>12} {'DoD':>8}")
    print("-" * 78)
    for r in sorted(all_results, key=lambda x: x['K']):
        print(f"{r['K']:>3} "
              f"{r['satisfaction_mean']:>10.4f}±{r['satisfaction_ci95']:<5.4f} "
              f"{r['cum_health_loss_mean']:>12.4e}±{r['cum_health_loss_ci95']:<.1e} "
              f"{r['completion_rate_mean']:>8.4f} "
              f"{r['avg_health_loss_mean']:>12.4e} "
              f"{r['avg_dod_mean']:>8.4f}")

    if not args.no_plots:
        try:
            plot_results(all_results, parent_dir)
        except Exception as e:
            print(f"[警告] 绘图失败：{e}")

    print(f"\n全部完成！结果目录：{parent_dir}")


if __name__ == '__main__':
    main()
