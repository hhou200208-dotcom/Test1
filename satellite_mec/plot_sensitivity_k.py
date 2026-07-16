"""plot_sensitivity_k.py
==========================
Plot K_MAX (multi-hop forwarding budget) sensitivity curves for the LyaMAPPO
paper, in the same paper-grade style as plot_sensitivity_v.py / plot_paper.py.

Reads <dir>/sensitivity_k_results.json (produced by sensitivity_k.py) and
generates paper-grade figures:

  26_k_satisfaction.png     User satisfaction (caliber A) vs K, with 95% CI
  27_k_cumulative_hl.png    Cumulative battery health loss vs K, with 95% CI
  28_k_sensitivity_panel.png  two-metric panel (satisfaction | health loss)
  29_k_pareto.png           satisfaction-vs-health-loss trade-off trace over K

Style conventions (matched to the paper's existing figures):
  - proposed method LyaMAPPO uses color 'C3', marker 'o' with black edge
  - xlabel/ylabel fontsize 12, bold title fontsize 13
  - dashed grid at alpha 0.4, dpi 150, English axis labels

Usage:
    python plot_sensitivity_k.py --dir docs/figures_sensitivity_k
    python plot_sensitivity_k.py --dir results/<TIMESTAMP>_SensitivityK
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_MAIN = 'C3'   # LyaMAPPO (proposed) — same as plot_paper.py


def _load(dir_):
    fp = os.path.join(dir_, 'sensitivity_k_results.json')
    data = json.load(open(fp, encoding='utf-8'))
    data = sorted(data, key=lambda r: r['K'])
    ks   = [r['K'] for r in data]
    sat  = [r['satisfaction_mean'] for r in data]
    sat_e = [r.get('satisfaction_ci95', 0.0) for r in data]
    hl   = [r['cum_health_loss_mean'] for r in data]
    hl_e = [r.get('cum_health_loss_ci95', 0.0) for r in data]
    return data, ks, sat, sat_e, hl, hl_e


def _style_ax(ax, xlabel, ylabel, title, ks):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(ks)
    ax.grid(alpha=0.4, linestyle='--')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='docs/figures_sensitivity_k',
                    help='目录，含 sensitivity_k_results.json，图也存于此')
    args = ap.parse_args()

    data, ks, sat, sat_e, hl, hl_e = _load(args.dir)
    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 26: User satisfaction vs K ───────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(ks, sat, yerr=sat_e, marker='o', color=C_MAIN, linewidth=2.4,
                markersize=10, markerfacecolor=C_MAIN, markeredgecolor='black',
                markeredgewidth=1.2, capsize=5, elinewidth=1.5,
                label='LyaMAPPO (proposed)', zorder=5)
    _style_ax(ax, 'Max forwarding hops $K$', 'User Satisfaction Rate',
              'User satisfaction sensitivity to $K$', ks)
    ax.legend(loc='best', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '26_k_satisfaction.png'), dpi=150)
    plt.close()
    print('saved 26_k_satisfaction.png')

    # ── Figure 27: Cumulative battery health loss vs K ──────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(ks, hl, yerr=hl_e, marker='o', color=C_MAIN, linewidth=2.4,
                markersize=10, markerfacecolor=C_MAIN, markeredgecolor='black',
                markeredgewidth=1.2, capsize=5, elinewidth=1.5,
                label='LyaMAPPO (proposed)', zorder=5)
    _style_ax(ax, 'Max forwarding hops $K$', 'Cumulative Battery Health Loss',
              'Battery life degradation sensitivity to $K$', ks)
    ax.legend(loc='best', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '27_k_cumulative_hl.png'), dpi=150)
    plt.close()
    print('saved 27_k_cumulative_hl.png')

    # ── Figure 28: two-metric panel ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].errorbar(ks, sat, yerr=sat_e, marker='o', color=C_MAIN, linewidth=2.4,
                     markersize=10, markerfacecolor=C_MAIN, markeredgecolor='black',
                     markeredgewidth=1.2, capsize=5, elinewidth=1.5, zorder=5)
    _style_ax(axes[0], 'Max forwarding hops $K$', 'User Satisfaction Rate',
              '(a) User satisfaction vs $K$', ks)
    axes[1].errorbar(ks, hl, yerr=hl_e, marker='o', color=C_MAIN, linewidth=2.4,
                     markersize=10, markerfacecolor=C_MAIN, markeredgecolor='black',
                     markeredgewidth=1.2, capsize=5, elinewidth=1.5, zorder=5)
    _style_ax(axes[1], 'Max forwarding hops $K$', 'Cumulative Battery Health Loss',
              '(b) Battery health loss vs $K$', ks)
    plt.suptitle('LyaMAPPO sensitivity to max forwarding hops $K$',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '28_k_sensitivity_panel.png'), dpi=150)
    plt.close()
    print('saved 28_k_sensitivity_panel.png')

    # ── Figure 29: satisfaction–health-loss trade-off trace ─────
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(sat, hl, '-', color=C_MAIN, linewidth=2.0, alpha=0.6, zorder=2)
    for r, sx, hy in zip(data, sat, hl):
        ax.scatter(sx, hy, s=180, marker='o', color=C_MAIN,
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(f'K={r["K"]}', (sx, hy), xytext=(8, 6),
                    textcoords='offset points', fontsize=10, fontweight='bold')
    ax.set_xlabel('User Satisfaction Rate', fontsize=12)
    ax.set_ylabel('Cumulative Battery Health Loss', fontsize=12)
    ax.set_title('Satisfaction–battery-life trade-off swept by $K$',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '29_k_pareto.png'), dpi=150)
    plt.close()
    print('saved 29_k_pareto.png')

    # summary
    print('\nK sweep summary:')
    print(f"  {'K':>3} {'Satisfaction':>14} {'CumHealthLoss':>16}")
    for r in data:
        print(f"  {r['K']:>3} {r['satisfaction_mean']:>14.4f} "
              f"{r['cum_health_loss_mean']:>16.4e}")


if __name__ == '__main__':
    main()
