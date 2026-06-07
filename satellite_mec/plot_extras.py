"""plot_extras.py
===================
Generate two additional paper-style PDF figures:

  11_system_delay_overhead.png
     Per-slot total queue backlog (QF + QB) distribution across eval slots
     for the 5 policies. Defined as "system delay overhead" in
     Zhang TMC 2023, since by Little's law the average sojourn time
     is proportional to the average queue length.

  12_satisfaction_pdf_only.png
     Per-slot user satisfaction rate distribution (single panel,
     drop-in replacement for the earlier 4-subplot version).

Both figures mimic Zhang TMC 2023's reference style: KDE-smoothed curves
with distinct markers every N points, grid, top-left legend.

Usage:
    python plot_extras.py --dir results/<TIMESTAMP>_MAPPO_lh4.0_final_b
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

POLICIES = ['MAPPO', 'LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']
COLORS   = {
    'MAPPO':          'C3',
    'MHSPO':          'C0',
    'LyapunovGreedy': 'C2',
    'GreedyDelay':    'C1',
    'LocalOnly':      'C7',
}
MARKERS  = {
    'MAPPO':          's',  # square
    'MHSPO':          'o',  # circle
    'LyapunovGreedy': '^',  # triangle
    'GreedyDelay':    'D',  # diamond
    'LocalOnly':      'x',  # x
}


def load_slot_csv(d, pol, key, scale=1.0):
    """Read float column from slot_metrics.csv where phase=='eval'."""
    p = os.path.join(d, pol, 'slot_metrics.csv')
    vals = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r['phase'] == 'eval':
                vals.append(float(r[key]) * scale)
    return np.array(vals)


def kde_with_markers(ax, samples, x_range, label, color, marker,
                     n_markers=15, lw=1.8):
    """KDE curve + sparse markers, paper style."""
    if len(samples) < 2 or samples.std() < 1e-9:
        return
    kde = gaussian_kde(samples)
    x = np.linspace(*x_range, 400)
    y = kde(x)
    # smooth line (no marker)
    ax.plot(x, y, color=color, linewidth=lw, alpha=0.85)
    # sparse markers on top
    idx = np.linspace(0, len(x) - 1, n_markers, dtype=int)
    ax.plot(x[idx], y[idx], color=color, marker=marker,
            markersize=7, linestyle='None', label=label,
            markerfacecolor=color, markeredgecolor='black',
            markeredgewidth=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    d = args.dir
    out_dir = args.out_dir or os.path.join(d, 'figures_final')
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 11: System delay overhead PDF (constellation-wide) ──
    # CSV 的 total_queue_size 是每星均值。25 颗星整体延迟开销：
    #   L_sys (tasks)  = N_SATS · Q_per_sat / S_avg
    #   λ_sys (tasks/s)= N_SATS · λ_per_sat
    #   W_per_task     = L_sys / λ_sys = Q_per_sat / S_avg / λ_per_sat  (与单星同)
    #   系统总开销     = L_sys · W_per_task = N_SATS · (Q_per_sat/S_avg) · W_per_task
    # 等效写法：N_SATS × per-sat sojourn time，单位 "task-seconds"
    S_AVG_MBIT = 30.0           # (S_MIN+S_MAX)/2 = (10+50)/2 Mbit
    LAMBDA_PER_SAT = 0.880       # 高/低混合到达率
    TAU = 1.0
    N_SATS = 25
    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_samples = {}
    for pol in POLICIES:
        q_mbit_per_sat = load_slot_csv(d, pol, 'total_queue_size', scale=1.0 / 1e6)
        per_sat_W = (q_mbit_per_sat / S_AVG_MBIT) / LAMBDA_PER_SAT * TAU  # s
        # 25 颗卫星整体：L_sys · W = N_SATS · (Q/S_avg) · W = task-seconds backlog
        L_sys_tasks = N_SATS * q_mbit_per_sat / S_AVG_MBIT
        sys_overhead = L_sys_tasks * per_sat_W                            # task·s
        all_samples[pol] = sys_overhead
    lo = min(s.min() for s in all_samples.values())
    hi = max(s.max() for s in all_samples.values())
    margin = (hi - lo) * 0.05
    x_range = (max(0, lo - margin), hi + margin)
    for pol in POLICIES:
        kde_with_markers(ax, all_samples[pol], x_range,
                         label=pol, color=COLORS[pol], marker=MARKERS[pol])
    ax.set_xlabel('System delay overhead (task·s, 25 satellites)', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '11_system_delay_overhead.png'), dpi=150)
    plt.close()
    print(f'saved 11_system_delay_overhead.png')

    # ── Figure 12: per-slot satisfaction rate PDF (single panel) ──
    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_samples = {}
    for pol in POLICIES:
        samples = load_slot_csv(d, pol, 'slot_satisfaction_rate')
        all_samples[pol] = samples
    x_range = (0.0, 1.05)
    for pol in POLICIES:
        kde_with_markers(ax, all_samples[pol], x_range,
                         label=pol, color=COLORS[pol], marker=MARKERS[pol])
    ax.set_xlabel('Per-slot satisfaction rate', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '12_satisfaction_pdf_only.png'), dpi=150)
    plt.close()
    print(f'saved 12_satisfaction_pdf_only.png')

    print(f'\nfigures saved to: {out_dir}')


if __name__ == '__main__':
    main()
