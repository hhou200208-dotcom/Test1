"""plot_sensitivity_v.py
==========================
Plot V sensitivity curves for the LyaMAPPO paper.

Reads results/<TIMESTAMP>_SensV_paper/sensitivity_v_results.json and
generates two paper-grade figures:

  22_v_sensitivity_panel.png    six metrics vs V on log-x; LyaMAPPO swept,
                                baselines as constant reference lines
  23_v_sensitivity_pareto.png   CR vs HL Pareto frontier as V varies

Usage:
    python plot_sensitivity_v.py --dir results/<TIMESTAMP>_SensV_paper
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLORS = {
    'MHSPO':          'C0',
    'LyapunovGreedy': 'C2',
    'GreedyDelay':    'C1',
    'LocalOnly':      'C7',
}
BASELINE_LS = {
    'MHSPO':          '--',
    'LyapunovGreedy': ':',
    'GreedyDelay':    '-.',
    'LocalOnly':      (0, (3, 1, 1, 1)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()

    fp = os.path.join(args.dir, 'sensitivity_v_results.json')
    data = json.load(open(fp))
    sweep    = data['mappo_sweep']
    baselines = data.get('baselines_reference', {})

    vs = [r['V'] for r in sweep]
    if len(vs) == 0:
        print('no V sweep results in JSON, abort')
        return

    out_dir = args.dir  # save into same dir
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ('CR',       'Completion Rate',     False),
        ('Sat',      'Satisfaction Rate',   False),
        ('delay',    'E2E Delay (s)',       False),
        ('HL',       'Health Loss / slot',  True),
        ('DoD',      'Average DoD',         False),
        ('queue_mb', 'Queue Backlog (MB)',  False),
    ]

    # ── Figure 22: 6-metric V sensitivity panel ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, label, log_y) in zip(axes.flat, metrics):
        # LyaMAPPO sweep curve
        ys = [r[key] for r in sweep]
        ax.plot(vs, ys, marker='o', color='C3', linewidth=2.2,
                markersize=8, markerfacecolor='C3', markeredgecolor='black',
                label='LyaMAPPO (swept)', zorder=5)
        # Baselines as constant horizontal reference
        for bname, bvals in baselines.items():
            yb = bvals.get(key, None)
            if yb is None: continue
            ax.axhline(yb, linestyle=BASELINE_LS[bname], color=BASELINE_COLORS[bname],
                       linewidth=1.5, alpha=0.85, label=bname)
        ax.set_xscale('log')
        if log_y: ax.set_yscale('log')
        ax.set_xlabel('V (Lyapunov trade-off)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.4, which='both', linestyle='--')
        if key == 'CR':  # legend only on first subplot
            ax.legend(loc='best', fontsize=8, ncol=2)
    plt.suptitle('LyaMAPPO V sensitivity vs constant baselines',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '22_v_sensitivity_panel.png'), dpi=150)
    plt.close()
    print('saved 22_v_sensitivity_panel.png')

    # ── Figure 23: CR–HL Pareto frontier ────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 6))
    crs = [r['CR'] for r in sweep]
    hls = [r['HL'] for r in sweep]
    # connect with line
    ax.plot(crs, hls, '-', color='C3', linewidth=2.0, alpha=0.6, zorder=2)
    for r, cr, hl in zip(sweep, crs, hls):
        ax.scatter(cr, hl, s=180, marker='o', color='C3',
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(f'V={r["V"]:g}', (cr, hl), xytext=(8, 6),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    # baselines as scatter points
    for bname, bvals in baselines.items():
        ax.scatter(bvals['CR'], bvals['HL'], s=150,
                   color=BASELINE_COLORS[bname], marker='s',
                   edgecolors='black', linewidths=1.2,
                   label=bname, zorder=4)
        ax.annotate(bname, (bvals['CR'], bvals['HL']), xytext=(8, -10),
                    textcoords='offset points', fontsize=9, color=BASELINE_COLORS[bname])
    ax.set_xlabel('Completion Rate', fontsize=12)
    ax.set_ylabel('Health Loss / slot', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('CR–HL Pareto frontier swept by V '
                 '(LyaMAPPO trace, baselines as squares)',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '23_v_sensitivity_pareto.png'), dpi=150)
    plt.close()
    print('saved 23_v_sensitivity_pareto.png')

    # Summary print
    print('\nV sweep summary:')
    print(f"  {'V':>8} {'CR':>9} {'HL':>13} {'DoD':>9}")
    for r in sweep:
        print(f"  {r['V']:>8.1f} {r['CR']:>9.4f} {r['HL']:>13.3e} {r['DoD']:>9.4f}")


if __name__ == '__main__':
    main()
