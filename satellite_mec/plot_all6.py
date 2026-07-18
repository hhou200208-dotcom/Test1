"""plot_all6.py
==================
6-policy comparison bar chart + CR-HL scatter:
  LyaMAPPO, MAPPO_NoBat, MHSPO, LyapunovGreedy, GreedyDelay, LocalOnly

Usage:
    python plot_all6.py \
        --full_dir results/<TIMESTAMP>_MAPPO_lh4.0_final_b \
        --nobat_dir results/<TIMESTAMP>_MAPPO_lh4.0_ablation_nobat
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Display ordering: best to worst CR, with ablation right after LyaMAPPO
POLICIES = ['LyaMAPPO', 'MAPPO_NoBat', 'MHSPO',
            'LyapunovGreedy', 'GreedyDelay', 'LocalOnly']

COLORS = {
    'LyaMAPPO':       'C3',   # red — proposed
    'MAPPO_NoBat':    'C8',   # gray — ablation
    'MHSPO':          'C0',   # blue — best baseline
    'LyapunovGreedy': 'C2',   # green
    'GreedyDelay':    'C1',   # orange
    'LocalOnly':      'C7',   # gray-dark
}
HATCH = {
    'LyaMAPPO':       '',
    'MAPPO_NoBat':    '///',  # ablation hatched
    'MHSPO':          '',
    'LyapunovGreedy': '',
    'GreedyDelay':    '',
    'LocalOnly':      '',
}
MARKERS = {
    'LyaMAPPO':       '*',
    'MAPPO_NoBat':    'D',
    'MHSPO':          'o',
    'LyapunovGreedy': '^',
    'GreedyDelay':    's',
    'LocalOnly':      'X',
}


def load_eval(d, pol_dir_name):
    p = os.path.join(d, pol_dir_name, 'eval_runs.json')
    return json.load(open(p))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_dir', required=True)
    ap.add_argument('--nobat_dir', required=True)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.full_dir, 'figures_final')
    os.makedirs(out_dir, exist_ok=True)

    # Map display name → (results dir, on-disk policy folder)
    data = {
        'LyaMAPPO':       load_eval(args.full_dir,  'MAPPO'),
        'MAPPO_NoBat':    load_eval(args.nobat_dir, 'MAPPO_NoBat'),
        'MHSPO':          load_eval(args.full_dir,  'MHSPO'),
        'LyapunovGreedy': load_eval(args.full_dir,  'LyapunovGreedy'),
        'GreedyDelay':    load_eval(args.full_dir,  'GreedyDelay'),
        'LocalOnly':      load_eval(args.full_dir,  'LocalOnly'),
    }

    metrics = [
        ('completion_rate',       'Completion Rate',     False),
        ('avg_satisfaction_rate', 'Satisfaction Rate',   False),
        ('avg_e2e_delay',         'E2E Delay (s)',       False),
        ('avg_health_loss',       'Health Loss / slot',  True),
        ('avg_dod',               'Average DoD',         False),
        ('avg_queue_mb',          'Queue Backlog (MB)',  False),
    ]

    # ── Figure 15: All-6 bar chart ──────────────────────────────
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

    plt.suptitle('Six-policy comparison at λ_high = 4.0 '
                 '(LyaMAPPO = proposed; MAPPO_NoBat = ablation)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '15_all6_bar.png'), dpi=150)
    plt.close()
    print('saved 15_all6_bar.png')

    # ── Figure 16: All-6 CR-HL scatter ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for name in POLICIES:
        d_pol = data[name]
        cr = d_pol['completion_rate']
        hl = d_pol['avg_health_loss']
        size = 320 if name == 'LyaMAPPO' else 160
        edge = 'red' if name == 'LyaMAPPO' else 'black'
        ax.scatter(cr, hl, s=size, marker=MARKERS[name],
                   color=COLORS[name],
                   edgecolors=edge, linewidths=1.5,
                   label=name, zorder=5)
        ax.annotate(name, (cr, hl), xytext=(8, 8),
                    textcoords='offset points', fontsize=10,
                    fontweight='bold' if name == 'LyaMAPPO' else 'normal')

    ax.set_xlabel('Completion Rate (CR)', fontsize=12)
    ax.set_ylabel('Health Loss / slot', fontsize=12)
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_title('CR vs HL Trade-off — six policies '
                 '(lower-right is better)',
                 fontsize=13, fontweight='bold')

    # Ablation arrow: NoBat → LyaMAPPO
    p1 = (data['MAPPO_NoBat']['completion_rate'], data['MAPPO_NoBat']['avg_health_loss'])
    p2 = (data['LyaMAPPO']['completion_rate'],     data['LyaMAPPO']['avg_health_loss'])
    ax.annotate('', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.7))
    midx = (p1[0] + p2[0]) / 2
    midy = np.sqrt(p1[1] * p2[1])
    ax.annotate('+battery\nreward',
                (midx, midy), xytext=(-55, -10),
                textcoords='offset points',
                fontsize=10, color='red', fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '16_all6_cr_hl_scatter.png'), dpi=150)
    plt.close()
    print('saved 16_all6_cr_hl_scatter.png')

    # ── Print full summary table ────────────────────────────────
    print('\n' + '=' * 80)
    print(f"  {'Policy':<16} {'CR':>8} {'Sat':>8} {'Delay':>8} "
          f"{'HL':>11} {'DoD':>8} {'Queue':>8}")
    print('-' * 80)
    for name in POLICIES:
        d_pol = data[name]
        print(f"  {name:<16} {d_pol['completion_rate']:>8.4f} "
              f"{d_pol.get('avg_satisfaction_rate', 0):>8.4f} "
              f"{d_pol.get('avg_e2e_delay', 0):>8.3f} "
              f"{d_pol['avg_health_loss']:>11.3e} "
              f"{d_pol['avg_dod']:>8.4f} "
              f"{d_pol.get('avg_queue_mb', 0):>8.2f}")
    print('=' * 80)


if __name__ == '__main__':
    main()
