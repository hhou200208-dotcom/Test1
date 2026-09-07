"""plot_final.py
==================
从 results/<final_dir> 提取 5 项指标 + 学习曲线，画论文级图：
  1. 5 项指标 bar / radar chart 对比 5 策略
  2. CR over time（eval 阶段每 slot 累计 CR）
  3. DoD over time
  4. HL over time（累积）
  5. 满意度 over time
  6. 队列积压 over time
  7. e2e delay PDF
  8. MAPPO 训练学习曲线

用法:
    python plot_final.py --dir results/<TIMESTAMP>_MAPPO_lh4.0_final_b
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POLICIES = ['MAPPO', 'LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']
COLORS   = {
    'MAPPO':          'C3',
    'MHSPO':          'C0',
    'LyapunovGreedy': 'C2',
    'GreedyDelay':    'C1',
    'LocalOnly':      'C7',
}
LINESTYLES = {
    'MAPPO': '-',
    'MHSPO': '--',
    'LyapunovGreedy': ':',
    'GreedyDelay': '-.',
    'LocalOnly': '-',
}


def load_curves(d, pol):
    """Load eval curve data for policy."""
    p = os.path.join(d, pol, 'curves.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_eval_run(d, pol):
    p = os.path.join(d, pol, 'eval_runs.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))[0]


def load_learning_curve(d):
    p = os.path.join(d, 'MAPPO', 'model', 'learning_curve.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def plot_curve(ax, curves_by_pol, x_key, y_key, title, ylabel, scale=1.0):
    """Generic per-slot curve plot."""
    for pol in POLICIES:
        if curves_by_pol.get(pol) is None:
            continue
        c = curves_by_pol[pol]
        # find the curve dict containing y_key
        for sub_key, sub_dict in c.items():
            if y_key in sub_dict:
                x = sub_dict[x_key]
                y = [v * scale for v in sub_dict[y_key]]
                ax.plot(x, y, label=pol, color=COLORS[pol],
                        linestyle=LINESTYLES[pol], linewidth=1.8,
                        alpha=0.85)
                break
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Slot')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    d = args.dir

    out_dir = os.path.join(d, 'figures_final')
    os.makedirs(out_dir, exist_ok=True)

    curves_by_pol = {p: load_curves(d, p) for p in POLICIES}
    eval_runs     = {p: load_eval_run(d, p) for p in POLICIES}
    learning      = load_learning_curve(d)

    # ── Figure 1: 5 metrics bar chart ──────────────────────────
    metrics = ['completion_rate', 'avg_satisfaction_rate',
               'avg_e2e_delay', 'avg_health_loss', 'avg_dod', 'avg_queue_mb']
    metric_labels = ['Completion Rate', 'Satisfaction Rate',
                     'E2E Delay (s)', 'Health Loss / slot',
                     'Average DoD', 'Queue Backlog (MB)']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, m, lab in zip(axes.flat, metrics, metric_labels):
        vals  = []
        names = []
        for pol in POLICIES:
            if eval_runs.get(pol) is None:
                continue
            v = eval_runs[pol].get(m, 0)
            vals.append(v); names.append(pol)
        x = np.arange(len(names))
        colors = [COLORS[n] for n in names]
        bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, fontsize=9)
        ax.set_title(lab, fontsize=12, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f'{v:.3g}', ha='center', va='bottom', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        if m == 'avg_health_loss':
            ax.set_yscale('log')
    plt.suptitle('Five-metric Comparison Across 5 Policies (λ_high=4.0)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_5_metrics_bar.png'), dpi=150)
    plt.close()
    print(f'saved 01_5_metrics_bar.png')

    # ── Figure 2: CR over time ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'completion_rate',
               'Cumulative Completion Rate over Slots',
               'Completion Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_cr_curve.png'), dpi=150)
    plt.close()
    print(f'saved 02_cr_curve.png')

    # ── Figure 3: DoD over time ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'avg_dod',
               'Average Depth of Discharge over Slots',
               'Average DoD')
    ax.axhspan(0.20, 0.40, color='green', alpha=0.1,
               label='Paper target range')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_dod_curve.png'), dpi=150)
    plt.close()
    print(f'saved 03_dod_curve.png')

    # ── Figure 4: HL cumulative ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'cumulative_health_loss',
               'Cumulative Battery Health Loss over Slots',
               'Cumulative HL')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_hl_cumulative.png'), dpi=150)
    plt.close()
    print(f'saved 04_hl_cumulative.png')

    # ── Figure 5: satisfaction over time ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'slot_satisfaction_rate',
               'Per-Slot User Satisfaction Rate over Slots',
               'Satisfaction Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_satisfaction_curve.png'), dpi=150)
    plt.close()
    print(f'saved 05_satisfaction_curve.png')

    # ── Figure 6: queue backlog over time ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for pol in POLICIES:
        if curves_by_pol.get(pol) is None: continue
        c = curves_by_pol[pol]['queue_curve']
        x = c['slots']
        y = [(a + b) / 1e6 for a, b in zip(c['avg_qf_size'], c['avg_qb_size'])]
        ax.plot(x, y, label=pol, color=COLORS[pol],
                linestyle=LINESTYLES[pol], linewidth=1.8, alpha=0.85)
    ax.set_title('Queue Backlog (QF + QB) over Slots')
    ax.set_ylabel('Queue backlog (MB)')
    ax.set_xlabel('Slot')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_queue_curve.png'), dpi=150)
    plt.close()
    print(f'saved 06_queue_curve.png')

    # ── Figure 7: MAPPO learning curve ───────────────────────────
    if learning:
        updates = [r['update']          for r in learning]
        crs     = [r['completion_rate'] for r in learning]
        dods    = [r['avg_dod']         for r in learning]
        hls     = [r['avg_health_loss'] for r in learning]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(updates, crs, 'C3-o', markersize=3, linewidth=1.5)
        axes[0].set_title('MAPPO Training CR')
        axes[0].set_xlabel('PPO Update'); axes[0].set_ylabel('Completion Rate')
        axes[0].grid(alpha=0.3)

        axes[1].plot(updates, dods, 'C2-o', markersize=3, linewidth=1.5)
        axes[1].set_title('MAPPO Training DoD')
        axes[1].set_xlabel('PPO Update'); axes[1].set_ylabel('Average DoD')
        axes[1].axhspan(0.20, 0.40, color='green', alpha=0.1)
        axes[1].grid(alpha=0.3)

        axes[2].plot(updates, hls, 'C0-o', markersize=3, linewidth=1.5)
        axes[2].set_title('MAPPO Training Health Loss')
        axes[2].set_xlabel('PPO Update'); axes[2].set_ylabel('HL / slot')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '07_mappo_learning_curve.png'),
                    dpi=150)
        plt.close()
        print(f'saved 07_mappo_learning_curve.png')

    # ── Figure 8: radar/spider chart ─────────────────────────────
    metrics_norm = {
        'CR':              ('completion_rate',         'max_better'),
        'Satisfaction':    ('avg_satisfaction_rate',   'max_better'),
        '1/Delay':         ('avg_e2e_delay',           'min_better'),
        '1/HL':            ('avg_health_loss',         'min_better'),
        '1/Queue':         ('avg_queue_mb',            'min_better'),
        'DoD near 0.3':    ('avg_dod',                 'target_0.3'),
    }
    raw = {m: [eval_runs[p].get(k, 0) for p in POLICIES if eval_runs.get(p)]
           for m, (k, _) in metrics_norm.items()}
    norm = {}
    for m, (k, mode) in metrics_norm.items():
        vals = np.array(raw[m], dtype=float)
        if mode == 'max_better':
            norm[m] = vals / max(vals.max(), 1e-12)
        elif mode == 'min_better':
            norm[m] = vals.min() / np.maximum(vals, 1e-12)
        elif mode == 'target_0.3':
            d = np.abs(vals - 0.3)
            norm[m] = 1.0 - d / max(d.max(), 1e-12)

    labels = list(metrics_norm.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    avail_pols = [p for p in POLICIES if eval_runs.get(p) is not None]
    for i, pol in enumerate(avail_pols):
        values = [norm[m][i] for m in labels] + [norm[labels[0]][i]]
        ax.plot(angles, values, color=COLORS[pol], linewidth=2,
                label=pol, marker='o', markersize=4)
        ax.fill(angles, values, color=COLORS[pol], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('Overall Performance Radar (normalized, larger is better)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '08_radar_overall.png'), dpi=150)
    plt.close()
    print(f'saved 08_radar_overall.png')

    print(f'\nall figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
