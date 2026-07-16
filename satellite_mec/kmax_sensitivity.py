"""
kmax_sensitivity.py
===================
最大转发跳数 K_MAX 敏感性实验。

目的
----
逐一评估 K_MAX ∈ {1,2,3,4,5} 下系统的“协同收益”与“中继开销”，
用于论证主实验取 K_MAX=3 的折中依据（回答审稿人“为什么是 3 跳”）：
  - 协同收益：完成率、用户满意度、（下降的）端到端时延；
  - 中继开销：累计电池寿命损耗、传输侧寿命损耗、逐时隙转发量、队列积压。
典型预期：1→3 跳收益快速改善，>3 跳收益饱和甚至恶化（时延/传输能耗/DoD 上升）。

策略选择
--------
默认 LyapunovGreedy —— 它按每时隙 Lyapunov 边费用（背压 + 电池惩罚）贪心决策，
与本文方法的决策规则同源，且不需训练、随 K_MAX 直接变化，最适合做跳数敏感性扫描。
也可选 mhspo / greedy / local，或 mappo（需 --train，逐 K 训练，较慢）。

用法
----
    python kmax_sensitivity.py                          # 默认 LyapunovGreedy，全量
    python kmax_sensitivity.py --debug                  # 快速冒烟（小时长/单种子）
    python kmax_sensitivity.py --policy mhspo
    python kmax_sensitivity.py --kmax 1 2 3 4 5 --n_runs 5 --t_eval 5400
    python kmax_sensitivity.py --policy mappo --train --t_train 20000   # 逐 K 训练 MAPPO（慢）

输出
----
    results/<时间戳>_KmaxSensitivity/
        kmax_summary.json     # 每个 K 的各指标 mean/std（跨种子）
        kmax_summary.csv      # 同上，便于直接贴表
        figures/kmax_overview.png            # 2x3 总览图
        figures/kmax_<metric>.png            # 各指标单图
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

from core import Config, SatelliteMECEnv
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)


# ──────────────────────────────────────────────────────────────
# 策略工厂
# ──────────────────────────────────────────────────────────────
def build_policy(name: str, cfg: Config, env: SatelliteMECEnv):
    name = name.lower()
    if name in ('lyapunov_greedy', 'lyagreedy', 'lyapunovgreedy'):
        return LyapunovGreedyPolicy(cfg, env)
    if name == 'mhspo':
        return MHSPOPolicy(cfg, env)
    if name in ('greedy', 'greedydelay'):
        return GreedyDelayPolicy(cfg, env)
    if name in ('local', 'localonly', 'lso'):
        return LocalOnlyPolicy(cfg, env)
    if name == 'mappo':
        from training import MAPPOPolicy
        return MAPPOPolicy(cfg, name='MAPPO')
    raise ValueError(f"未知策略：{name}")


# ──────────────────────────────────────────────────────────────
# 单次评估（一个 K、一个种子）
# ──────────────────────────────────────────────────────────────
def evaluate_once(cfg: Config, policy, env: SatelliteMECEnv,
                  seeds: Dict[str, int], t_eval: int,
                  warmup_frac: float = 0.1) -> Dict[str, float]:
    """在给定种子上评估策略，返回该次运行的各项汇总指标。"""
    env.reset(phase='eval', seeds=seeds)
    policy.set_eval_mode()

    delays: List[float] = []
    hl_cum = 0.0            # 累计电池寿命损耗
    trans_hl_cum = 0.0      # 累计传输侧寿命损耗
    fwd_per_slot: List[int] = []
    queue_per_slot: List[float] = []
    warmup_slots = int(t_eval * warmup_frac)

    last_info: Dict = {}
    for t in range(t_eval):
        _, _, _, info = env.step(policy=policy)
        last_info = info
        if t < warmup_slots:            # 丢弃前段暂态，仅统计稳态
            continue
        delays.extend(info.get('slot_e2e_delays', []))
        hl_cum       += info.get('avg_health_loss', 0.0)
        trans_hl_cum += info.get('avg_delta_l_trans', 0.0)
        fwd_per_slot.append(info.get('forwarded', 0))
        queue_per_slot.append(info.get('total_queue_size', 0.0))

    delays = np.asarray(delays, dtype=float)
    return {
        'completion_rate':        float(env.get_eval_completion_rate()),
        'satisfaction_rate':      float(last_info.get('eval_satisfaction_rate', 0.0)),
        'mean_delay':             float(delays.mean()) if delays.size else 0.0,
        'p95_delay':              float(np.percentile(delays, 95)) if delays.size else 0.0,
        'cumulative_health_loss': float(hl_cum),
        'trans_health_loss':      float(trans_hl_cum),
        'avg_forwarded':          float(np.mean(fwd_per_slot)) if fwd_per_slot else 0.0,
        'avg_total_queue':        float(np.mean(queue_per_slot)) if queue_per_slot else 0.0,
    }


# ──────────────────────────────────────────────────────────────
# 逐 K 扫描
# ──────────────────────────────────────────────────────────────
METRICS = [
    ('completion_rate',        'Task Completion Rate',        'higher_better'),
    ('satisfaction_rate',      'User Satisfaction Rate',      'higher_better'),
    ('mean_delay',             'Mean End-to-End Delay (s)',   'lower_better'),
    ('cumulative_health_loss', 'Cumulative Battery Loss',     'lower_better'),
    ('trans_health_loss',      'Transmission-Induced Loss',   'lower_better'),
    ('avg_forwarded',          'Forwarded Tasks / Slot',      'neutral'),
]


def run_sweep(args) -> Dict:
    results: Dict[int, Dict] = {}
    for k in args.kmax:
        cfg = Config()
        cfg.K_MAX = int(k)
        if args.n_runs is not None:
            cfg.N_EVAL_RUNS = args.n_runs
        t_eval  = args.t_eval  if args.t_eval  is not None else cfg.T_EVAL
        t_train = args.t_train if args.t_train is not None else cfg.T_TRAIN

        env = SatelliteMECEnv(cfg)
        policy = build_policy(args.policy, cfg, env)

        # 可选：MAPPO 需先训练（逐 K 各训一个策略）
        if args.policy.lower() == 'mappo' and args.train:
            print(f"[K={k}] 训练 MAPPO {t_train} 时隙 …")
            policy.set_train_mode(); env.reset(phase='train')
            for _ in range(t_train):
                policy.run_step(env)

        per_run: List[Dict[str, float]] = []
        n_runs = cfg.N_EVAL_RUNS if args.n_runs is None else args.n_runs
        for run_idx in range(n_runs):
            seeds = cfg.get_eval_seeds(run_idx)
            m = evaluate_once(cfg, policy, env, seeds, t_eval, args.warmup_frac)
            per_run.append(m)
            print(f"[K={k}] run{run_idx}: CR={m['completion_rate']:.3f} "
                  f"delay={m['mean_delay']:.2f}s HL={m['cumulative_health_loss']:.1f} "
                  f"fwd/slot={m['avg_forwarded']:.2f}")

        # 跨种子聚合 mean/std
        agg: Dict[str, Dict[str, float]] = {}
        for key in per_run[0]:
            vals = np.asarray([r[key] for r in per_run], dtype=float)
            agg[key] = {'mean': float(vals.mean()),
                        'std':  float(vals.std()),
                        'values': vals.tolist()}
        results[int(k)] = agg
    return results


# ──────────────────────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────────────────────
def plot_combined(results: Dict, out_dir: str, highlight_k: int = 3) -> None:
    """满意度（收益，左轴）与累计电池损耗（成本，右轴）合成一张双 Y 轴权衡图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ks = sorted(results.keys())
    os.makedirs(out_dir, exist_ok=True)
    sat_m = [results[k]['satisfaction_rate']['mean'] for k in ks]
    sat_s = [results[k]['satisfaction_rate']['std']  for k in ks]
    hl_m  = [results[k]['cumulative_health_loss']['mean'] for k in ks]
    hl_s  = [results[k]['cumulative_health_loss']['std']  for k in ks]

    c_sat, c_hl = '#1f77b4', '#d62728'
    fig, ax1 = plt.subplots(figsize=(6.2, 4.3))

    ax1.errorbar(ks, sat_m, yerr=sat_s, marker='o', color=c_sat, lw=2,
                 capsize=4, label='User Satisfaction Rate')
    ax1.set_xlabel(r'Max Forwarding Hops $K_{\max}$')
    ax1.set_ylabel('User Satisfaction Rate (higher better)', color=c_sat)
    ax1.tick_params(axis='y', labelcolor=c_sat)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.errorbar(ks, hl_m, yerr=hl_s, marker='s', color=c_hl, lw=2, ls='--',
                 capsize=4, label='Cumulative Battery Loss')
    ax2.set_ylabel('Cumulative Battery Loss (lower better)', color=c_hl)
    ax2.tick_params(axis='y', labelcolor=c_hl)

    if highlight_k in ks:
        ax1.axvline(highlight_k, color='gray', ls=':', alpha=0.8)
        ax1.annotate(f'$K_{{\\max}}={highlight_k}$\n(chosen trade-off)',
                     xy=(highlight_k, sat_m[ks.index(highlight_k)]),
                     xytext=(8, -28), textcoords='offset points',
                     color='dimgray', fontsize=9)

    # 合并两轴图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'kmax_satisfaction_vs_loss.png'), dpi=150)
    plt.close(fig)


def plot_results(results: Dict, out_dir: str, highlight_k: int = 3) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ks = sorted(results.keys())
    os.makedirs(out_dir, exist_ok=True)

    def series(metric):
        mean = np.array([results[k][metric]['mean'] for k in ks])
        std  = np.array([results[k][metric]['std']  for k in ks])
        return mean, std

    # 总览 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, ylabel, _kind) in zip(axes.ravel(), METRICS):
        mean, std = series(metric)
        ax.errorbar(ks, mean, yerr=std, marker='o', capsize=4, lw=2)
        if highlight_k in ks:
            ax.axvline(highlight_k, color='crimson', ls='--', alpha=0.6)
            ax.annotate(f'$K_{{\\max}}={highlight_k}$', xy=(highlight_k, mean[ks.index(highlight_k)]),
                        xytext=(6, 6), textcoords='offset points', color='crimson', fontsize=9)
        ax.set_xlabel(r'Max Forwarding Hops $K_{\max}$')
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Sensitivity to Maximum Forwarding Hops', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, 'kmax_overview.png'), dpi=150)
    plt.close(fig)

    # 单指标图
    for metric, ylabel, _kind in METRICS:
        mean, std = series(metric)
        f, ax = plt.subplots(figsize=(5.2, 4))
        ax.errorbar(ks, mean, yerr=std, marker='o', capsize=4, lw=2, color='#1f77b4')
        if highlight_k in ks:
            ax.axvline(highlight_k, color='crimson', ls='--', alpha=0.6)
        ax.set_xlabel(r'Max Forwarding Hops $K_{\max}$')
        ax.set_ylabel(ylabel)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        f.tight_layout()
        f.savefig(os.path.join(out_dir, f'kmax_{metric}.png'), dpi=150)
        plt.close(f)


# ──────────────────────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────────────────────
def save_results(results: Dict, base_dir: str) -> None:
    with open(os.path.join(base_dir, 'kmax_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    ks = sorted(results.keys())
    metrics = list(results[ks[0]].keys())
    with open(os.path.join(base_dir, 'kmax_summary.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['K_MAX'] + [f'{m}_mean' for m in metrics] + [f'{m}_std' for m in metrics]
        w.writerow(header)
        for k in ks:
            row = [k] + [results[k][m]['mean'] for m in metrics] \
                      + [results[k][m]['std'] for m in metrics]
            w.writerow(row)


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='K_MAX 跳数敏感性实验')
    p.add_argument('--policy', default='lyapunov_greedy',
                   help='lyapunov_greedy(默认) | mhspo | greedy | local | mappo')
    p.add_argument('--kmax', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                   help='要扫描的 K_MAX 列表')
    p.add_argument('--n_runs', type=int, default=None, help='评估种子数（默认取 Config）')
    p.add_argument('--t_eval', type=int, default=None, help='每次评估时隙数（默认取 Config）')
    p.add_argument('--t_train', type=int, default=None, help='mappo 训练时隙数')
    p.add_argument('--train', action='store_true', help='mappo 模式下逐 K 训练')
    p.add_argument('--warmup_frac', type=float, default=0.1,
                   help='评估前段丢弃比例（去暂态）')
    p.add_argument('--debug', action='store_true', help='快速冒烟：单种子、短时长')
    p.add_argument('--no_plots', action='store_true')
    p.add_argument('--only_combined', action='store_true',
                   help='只出满意度 vs 电池损耗合成图，不出其余单图')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        args.n_runs = 1
        args.t_eval = args.t_eval or 600
        args.t_train = args.t_train or 2000
        print("[DEBUG] 单种子、短时长快速冒烟")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', f'{ts}_KmaxSensitivity')
    os.makedirs(base_dir, exist_ok=True)
    print(f"[Kmax] 策略={args.policy}  K={args.kmax}  输出={base_dir}")

    t0 = time.time()
    results = run_sweep(args)
    save_results(results, base_dir)
    if not args.no_plots:
        fig_dir = os.path.join(base_dir, 'figures')
        hk = 3 if 3 in args.kmax else args.kmax[len(args.kmax) // 2]
        plot_combined(results, fig_dir, highlight_k=hk)   # 满意度 vs 电池损耗 合成图
        if not args.only_combined:
            plot_results(results, fig_dir, highlight_k=hk)

    # 控制台小结
    ks = sorted(results.keys())
    print("\n===== K_MAX 敏感性小结 =====")
    print(f"{'K':>3} | {'完成率':>7} | {'满意度':>7} | {'时延(s)':>8} | "
          f"{'累计HL':>9} | {'转发/slot':>9}")
    for k in ks:
        r = results[k]
        print(f"{k:>3} | {r['completion_rate']['mean']:>7.3f} | "
              f"{r['satisfaction_rate']['mean']:>7.3f} | "
              f"{r['mean_delay']['mean']:>8.2f} | "
              f"{r['cumulative_health_loss']['mean']:>9.1f} | "
              f"{r['avg_forwarded']['mean']:>9.2f}")
    print(f"\n完成，用时 {time.time() - t0:.1f}s，结果目录：{base_dir}")


if __name__ == '__main__':
    main()
