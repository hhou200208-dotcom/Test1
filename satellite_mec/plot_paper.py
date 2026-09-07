"""plot_paper.py
==================
Generate 5 paper-grade figures for the 6-policy comparison
(LyaMAPPO, MAPPO_NoBat, MHSPO, LyapunovGreedy, GreedyDelay, LocalOnly).

Removes any λ_high mentions from titles for cleaner paper layout.

  17_satisfaction_pdf.png      User satisfaction PDF (per-slot rate)
  18_delay_overhead_pdf.png    System delay overhead PDF (task·s)
  19_health_loss_curve.png     Cumulative HL over time
  20_queue_backlog_curve.png   Queue (QF+QB) over time
  21_six_metric_bar.png        6-metric bar chart × 6 policies

Usage:
    python plot_paper.py \
        --full_dir results/<TIMESTAMP>_MAPPO_lh4.0_final_b \
        --nobat_dir results/<TIMESTAMP>_MAPPO_lh4.0_ablation_nobat
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


POLICIES = ['LyaMAPPO', 'MAPPO_NoBat', 'MHSPO',
            'LyapunovGreedy', 'GreedyDelay', 'LocalOnly']

COLORS = {
    'LyaMAPPO':       'C3',
    'MAPPO_NoBat':    'C8',
    'MHSPO':          'C0',
    'LyapunovGreedy': 'C2',
    'GreedyDelay':    'C1',
    'LocalOnly':      'C7',
}
HATCH = {
    'LyaMAPPO':       '',
    'MAPPO_NoBat':    '///',
    'MHSPO':          '',
    'LyapunovGreedy': '',
    'GreedyDelay':    '',
    'LocalOnly':      '',
}
MARKERS = {
    'LyaMAPPO':       'o',
    'MAPPO_NoBat':    'D',
    'MHSPO':          's',
    'LyapunovGreedy': '^',
    'GreedyDelay':    'P',
    'LocalOnly':      'X',
}
LINESTYLES = {
    'LyaMAPPO':       '-',
    'MAPPO_NoBat':    '--',
    'MHSPO':          '-',
    'LyapunovGreedy': ':',
    'GreedyDelay':    '-.',
    'LocalOnly':      (0, (3, 1, 1, 1)),
}


# ── Constants for "system delay overhead" Little's-law conversion ─
S_AVG_MBIT     = 30.0
LAMBDA_PER_SAT = 0.880
N_SATS         = 25


def policy_to_dir(name):
    """Map display name → on-disk folder name."""
    if name == 'LyaMAPPO':    return 'MAPPO'
    if name == 'MAPPO_NoBat': return 'MAPPO_NoBat'
    return name


def load_slot_csv(d, pol, key):
    p = os.path.join(d, policy_to_dir(pol), 'slot_metrics.csv')
    vals = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r['phase'] == 'eval':
                vals.append(float(r[key]))
    return np.array(vals)


def load_eval(d, pol):
    p = os.path.join(d, policy_to_dir(pol), 'eval_runs.json')
    return json.load(open(p))[0]


def load_curves(d, pol):
    p = os.path.join(d, policy_to_dir(pol), 'curves.json')
    return json.load(open(p))


def kde_with_markers(ax, samples, x_range, label, color, marker, n_markers=15, lw=1.8):
    if len(samples) < 2 or samples.std() < 1e-9: return
    kde = gaussian_kde(samples)
    x = np.linspace(*x_range, 400)
    y = kde(x)
    ax.plot(x, y, color=color, linewidth=lw, alpha=0.85)
    idx = np.linspace(0, len(x) - 1, n_markers, dtype=int)
    ax.plot(x[idx], y[idx], color=color, marker=marker,
            markersize=7, linestyle='None', label=label,
            markerfacecolor=color, markeredgecolor='black',
            markeredgewidth=0.5)


def get_dir(name, args):
    return args.nobat_dir if name == 'MAPPO_NoBat' else args.full_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_dir',  required=True)
    ap.add_argument('--nobat_dir', required=True)
    ap.add_argument('--out_dir',   default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.full_dir, 'figures_paper')
    os.makedirs(out_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Figure 17: Per-slot satisfaction rate PDF (6 policies)
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_samples = {}
    for pol in POLICIES:
        s = load_slot_csv(get_dir(pol, args), pol, 'slot_satisfaction_rate')
        all_samples[pol] = s
    for pol in POLICIES:
        kde_with_markers(ax, all_samples[pol], (0, 1.05),
                         label=pol, color=COLORS[pol], marker=MARKERS[pol])
    ax.set_xlabel('Per-slot user satisfaction rate', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc='upper left', fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '17_satisfaction_pdf.png'), dpi=150)
    plt.close()
    print('saved 17_satisfaction_pdf.png')

    # ──────────────────────────────────────────────────────────
    # Figure 18: System delay overhead PDF (task·s, 25 sats)
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_samples = {}
    for pol in POLICIES:
        q_mbit_per_sat = load_slot_csv(get_dir(pol, args), pol, 'total_queue_size') / 1e6
        per_sat_W = (q_mbit_per_sat / S_AVG_MBIT) / LAMBDA_PER_SAT
        L_sys = N_SATS * q_mbit_per_sat / S_AVG_MBIT
        all_samples[pol] = L_sys * per_sat_W
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
    ax.legend(loc='upper right', fontsize=9, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '18_delay_overhead_pdf.png'), dpi=150)
    plt.close()
    print('saved 18_delay_overhead_pdf.png')

    # ──────────────────────────────────────────────────────────
    # Figure 19: Cumulative HL over time (6 policies)
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for pol in POLICIES:
        c = load_curves(get_dir(pol, args), pol)
        x = c['health_loss_curve']['slots']
        y = c['health_loss_curve']['cumulative_health_loss']
        ax.plot(x, y, label=pol, color=COLORS[pol],
                linestyle=LINESTYLES[pol], linewidth=2.0, alpha=0.85)
    ax.set_xlabel('Slot', fontsize=12)
    ax.set_ylabel('Cumulative battery health loss', fontsize=12)
    ax.set_title('Constellation-wide cumulative battery health loss',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '19_health_loss_curve.png'), dpi=150)
    plt.close()
    print('saved 19_health_loss_curve.png')

    # ──────────────────────────────────────────────────────────
    # Figure 20: Queue backlog over time (6 policies)
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for pol in POLICIES:
        c = load_curves(get_dir(pol, args), pol)
        x = c['queue_curve']['slots']
        y = [(a + b) / 1e6 for a, b in zip(c['queue_curve']['avg_qf_size'],
                                            c['queue_curve']['avg_qb_size'])]
        ax.plot(x, y, label=pol, color=COLORS[pol],
                linestyle=LINESTYLES[pol], linewidth=2.0, alpha=0.85)
    ax.set_xlabel('Slot', fontsize=12)
    ax.set_ylabel('Queue backlog (MB / satellite)', fontsize=12)
    ax.set_title('Per-satellite queue backlog (QF + QB)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc='best', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '20_queue_backlog_curve.png'), dpi=150)
    plt.close()
    print('saved 20_queue_backlog_curve.png')

    # ──────────────────────────────────────────────────────────
    # Figure 21: 6-metric bar chart × 6 policies
    # ──────────────────────────────────────────────────────────
    metrics = [
        ('completion_rate',       'Completion Rate',     False),
        ('avg_satisfaction_rate', 'Satisfaction Rate',   False),
        ('avg_e2e_delay',         'E2E Delay (s)',       False),
        ('avg_health_loss',       'Health Loss / slot',  True),
        ('avg_dod',               'Average DoD',         False),
        ('avg_queue_mb',          'Queue Backlog (MB)',  False),
    ]
    data = {pol: load_eval(get_dir(pol, args), pol) for pol in POLICIES}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(len(POLICIES))
    for ax, (key, label, log_y) in zip(axes.flat, metrics):
        vals = [data[p].get(key, 0) for p in POLICIES]
        bars = ax.bar(x, vals,
                      color=[COLORS[p] for p in POLICIES],
                      edgecolor='black', linewidth=1.0,
                      hatch=[HATCH[p] for p in POLICIES], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(POLICIES, fontsize=9, rotation=18, ha='right')
        ax.set_title(label, fontsize=12, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f'{v:.3g}', ha='center', va='bottom', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        if log_y:
            ax.set_yscale('log')
    plt.suptitle('Six-policy comparison (LyaMAPPO = proposed; MAPPO_NoBat = ablation)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '21_six_metric_bar.png'), dpi=150)
    plt.close()
    print('saved 21_six_metric_bar.png')

    print(f'\nall figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
