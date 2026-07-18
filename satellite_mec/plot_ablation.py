"""plot_ablation.py
====================
Ablation comparison figure: LyaMAPPO (full) vs MAPPO_NoBat (no battery in
reward) vs MHSPO (best baseline reference).

Shows the causal contribution of battery-aware Lyapunov virtual queue +
W_HL outcome term to the HL reduction.

Usage:
    python plot_ablation.py \
        --full_dir results/<timestamp>_MAPPO_lh4.0_final_b \
        --nobat_dir results/<timestamp>_MAPPO_lh4.0_ablation_nobat
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'LyaMAPPO':    'C3',     # red — proposed
    'MAPPO_NoBat': 'C8',     # gray-ish — ablation
    'MHSPO':       'C0',     # blue — baseline
}
HATCH = {
    'LyaMAPPO':    '',
    'MAPPO_NoBat': '///',    # hatched for ablation
    'MHSPO':       '',
}


def load_eval(d, pol):
    p = os.path.join(d, pol, 'eval_runs.json')
    return json.load(open(p))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_dir', required=True)
    ap.add_argument('--nobat_dir', required=True)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.full_dir, 'figures_final')
    os.makedirs(out_dir, exist_ok=True)

    data = {
        'LyaMAPPO':    load_eval(args.full_dir, 'MAPPO'),
        'MAPPO_NoBat': load_eval(args.nobat_dir, 'MAPPO_NoBat'),
        'MHSPO':       load_eval(args.full_dir, 'MHSPO'),
    }

    # ── Figure 13: Ablation 5-metric bar chart ──────────────────
    metrics = [
        ('completion_rate',       'Completion Rate',     'max_better', False),
        ('avg_satisfaction_rate', 'Satisfaction Rate',   'max_better', False),
        ('avg_e2e_delay',         'E2E Delay (s)',       'min_better', False),
        ('avg_health_loss',       'Health Loss / slot',  'min_better', True),
        ('avg_dod',               'Average DoD',         'closer_0.3', False),
        ('avg_queue_mb',          'Queue Backlog (MB)',  'min_better', False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    pol_names = list(data.keys())
    x = np.arange(len(pol_names))

    for ax, (key, label, mode, log_y) in zip(axes.flat, metrics):
        vals = [data[p].get(key, 0) for p in pol_names]
        bars = ax.bar(x, vals,
                      color=[COLORS[p] for p in pol_names],
                      edgecolor='black', linewidth=1.2,
                      hatch=[HATCH[p] for p in pol_names], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(pol_names, fontsize=10, rotation=12)
        ax.set_title(label, fontsize=12, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f'{v:.3g}', ha='center', va='bottom', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        if log_y:
            ax.set_yscale('log')

        # Highlight LyaMAPPO advantage with annotation arrow
        if key == 'avg_health_loss':
            v_full  = data['LyaMAPPO']['avg_health_loss']
            v_nobat = data['MAPPO_NoBat']['avg_health_loss']
            ratio = v_nobat / v_full
            ax.annotate(f'{ratio:.2f}× higher\nwithout battery\nmodeling',
                        xy=(1, v_nobat), xytext=(1.4, v_nobat * 0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=10, color='red', fontweight='bold')

    plt.suptitle(
        'Ablation: Effect of Lyapunov Battery-Aware Reward Components\n'
        '(LyaMAPPO − battery-in-reward = MAPPO_NoBat, λ_high=4.0)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '13_ablation_battery_reward.png'), dpi=150)
    plt.close()
    print(f'saved 13_ablation_battery_reward.png')

    # ── Figure 14: HL-CR scatter for paper headline ─────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    # Also include the other baselines for context
    full_dir = args.full_dir
    other_pols = ['LocalOnly', 'GreedyDelay', 'LyapunovGreedy']
    extra = {p: load_eval(full_dir, p) for p in other_pols}
    all_data = {**data, **extra}
    extra_colors = {'LocalOnly': 'C7', 'GreedyDelay': 'C1', 'LyapunovGreedy': 'C2'}
    extra_colors.update(COLORS)

    for name, d_pol in all_data.items():
        cr = d_pol['completion_rate']
        hl = d_pol['avg_health_loss']
        marker = 'D' if 'NoBat' in name else ('*' if name == 'LyaMAPPO' else 'o')
        size   = 250 if name == 'LyaMAPPO' else 140
        edge   = 'red' if name == 'LyaMAPPO' else 'black'
        ax.scatter(cr, hl, s=size, marker=marker,
                   color=extra_colors[name],
                   edgecolors=edge, linewidths=1.5, label=name, zorder=5)
        ax.annotate(name, (cr, hl), xytext=(7, 7),
                    textcoords='offset points', fontsize=10,
                    fontweight='bold' if name == 'LyaMAPPO' else 'normal')

    ax.set_xlabel('Completion Rate', fontsize=12)
    ax.set_ylabel('Health Loss / slot', fontsize=12)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_title('CR vs HL Trade-off (lower-right is better)',
                 fontsize=13, fontweight='bold')

    # Draw an arrow showing the ablation transition
    p1 = (data['MAPPO_NoBat']['completion_rate'], data['MAPPO_NoBat']['avg_health_loss'])
    p2 = (data['LyaMAPPO']['completion_rate'], data['LyaMAPPO']['avg_health_loss'])
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.7))
    midx, midy = (p1[0] + p2[0]) / 2, np.sqrt(p1[1] * p2[1])
    ax.annotate('+battery\nreward', (midx, midy), xytext=(-50, -10),
                textcoords='offset points', fontsize=10, color='red',
                fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '14_cr_hl_tradeoff.png'), dpi=150)
    plt.close()
    print(f'saved 14_cr_hl_tradeoff.png')

    # ── Print summary table ─────────────────────────────────────
    print('\n' + '=' * 78)
    print(f"  {'Variant':<14} {'CR':>8} {'Sat':>8} {'Delay':>8} "
          f"{'HL':>11} {'DoD':>8} {'Queue':>8}")
    print('-' * 78)
    for name in ['LyaMAPPO', 'MAPPO_NoBat', 'MHSPO']:
        d_pol = data[name]
        print(f"  {name:<14} {d_pol['completion_rate']:>8.4f} "
              f"{d_pol.get('avg_satisfaction_rate', 0):>8.4f} "
              f"{d_pol.get('avg_e2e_delay', 0):>8.3f} "
              f"{d_pol['avg_health_loss']:>11.3e} "
              f"{d_pol['avg_dod']:>8.4f} "
              f"{d_pol.get('avg_queue_mb', 0):>8.2f}")
    print('-' * 78)
    full_hl  = data['LyaMAPPO']['avg_health_loss']
    nobat_hl = data['MAPPO_NoBat']['avg_health_loss']
    mhspo_hl = data['MHSPO']['avg_health_loss']
    print(f'  HL ratio NoBat / LyaMAPPO = {nobat_hl / full_hl:.2f}×')
    print(f'  HL ratio MHSPO / LyaMAPPO = {mhspo_hl / full_hl:.2f}×')
    print(f"  → battery-aware reward causes {100 * (1 - full_hl/nobat_hl):.1f}% "
          f"HL reduction at similar CR")


if __name__ == '__main__':
    main()
