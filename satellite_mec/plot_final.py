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

    # ── Figure 1: 5 项指标 bar chart 对比 ──────────────────────
    metrics = ['completion_rate', 'avg_satisfaction_rate',
               'avg_e2e_delay', 'avg_health_loss', 'avg_dod', 'avg_queue_mb']
    metric_labels = ['CR', '满意度', '时延(s)', 'HL/slot', '平均DoD', '队列积压(MB)']

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
    plt.suptitle('λ_high=4.0 主场景下 5 策略 5 项指标对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_5_metrics_bar.png'), dpi=150)
    plt.close()
    print(f'保存 01_5_metrics_bar.png')

    # ── Figure 2: CR over time ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'completion_rate',
               '累计完成率 (CR) 随时隙演化', 'CR')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_cr_curve.png'), dpi=150)
    plt.close()
    print(f'保存 02_cr_curve.png')

    # ── Figure 3: DoD over time ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'avg_dod',
               '平均放电深度 (DoD) 随时隙演化', 'DoD')
    ax.axhspan(0.20, 0.40, color='green', alpha=0.1, label='论文目标区间')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_dod_curve.png'), dpi=150)
    plt.close()
    print(f'保存 03_dod_curve.png')

    # ── Figure 4: HL cumulative ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'cumulative_health_loss',
               '累积健康损失 (HL) 随时隙演化', 'Cumulative HL')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_hl_cumulative.png'), dpi=150)
    plt.close()
    print(f'保存 04_hl_cumulative.png')

    # ── Figure 5: 满意度 over time ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_curve(ax, curves_by_pol, 'slots', 'slot_satisfaction_rate',
               '每时隙满意度演化', 'Satisfaction rate')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '05_satisfaction_curve.png'), dpi=150)
    plt.close()
    print(f'保存 05_satisfaction_curve.png')

    # ── Figure 6: 队列积压 over time ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for pol in POLICIES:
        if curves_by_pol.get(pol) is None: continue
        c = curves_by_pol[pol]['queue_curve']
        x = c['slots']
        y = [(a + b) / 1e6 for a, b in zip(c['avg_qf_size'], c['avg_qb_size'])]
        ax.plot(x, y, label=pol, color=COLORS[pol],
                linestyle=LINESTYLES[pol], linewidth=1.8, alpha=0.85)
    ax.set_title('队列积压 (QF + QB) 随时隙演化')
    ax.set_ylabel('Queue backlog (MB)')
    ax.set_xlabel('Slot')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_queue_curve.png'), dpi=150)
    plt.close()
    print(f'保存 06_queue_curve.png')

    # ── Figure 7: MAPPO 学习曲线（如有）──────────────────────────
    if learning:
        updates = [r['update']          for r in learning]
        crs     = [r['completion_rate'] for r in learning]
        dods    = [r['avg_dod']         for r in learning]
        hls     = [r['avg_health_loss'] for r in learning]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(updates, crs, 'C3-o', markersize=3, linewidth=1.5)
        axes[0].set_title('MAPPO 训练 CR 演化')
        axes[0].set_xlabel('PPO Update'); axes[0].set_ylabel('CR')
        axes[0].grid(alpha=0.3)

        axes[1].plot(updates, dods, 'C2-o', markersize=3, linewidth=1.5)
        axes[1].set_title('MAPPO 训练 DoD 演化')
        axes[1].set_xlabel('PPO Update'); axes[1].set_ylabel('DoD')
        axes[1].axhspan(0.20, 0.40, color='green', alpha=0.1)
        axes[1].grid(alpha=0.3)

        axes[2].plot(updates, hls, 'C0-o', markersize=3, linewidth=1.5)
        axes[2].set_title('MAPPO 训练 HL 演化')
        axes[2].set_xlabel('PPO Update'); axes[2].set_ylabel('HL/slot')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '07_mappo_learning_curve.png'), dpi=150)
        plt.close()
        print(f'保存 07_mappo_learning_curve.png')

    # ── Figure 8: 5 项指标 radar/spider chart ────────────────────
    # 归一化每个指标（越大越好的归一化为 v/max；越小越好的归一化为 min/v）
    metrics_norm = {
        'CR':         ('completion_rate',         'max_better'),
        '满意度':     ('avg_satisfaction_rate',   'max_better'),
        '时延倒数':   ('avg_e2e_delay',           'min_better'),
        'HL 倒数':    ('avg_health_loss',         'min_better'),
        '队列倒数':   ('avg_queue_mb',            'min_better'),
        'DoD 接近目标': ('avg_dod',               'target_0.3'),
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
            # 越接近 0.3 分越高
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
    ax.set_title('5 策略综合性能雷达图 (归一化, 越大越好)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '08_radar_overall.png'), dpi=150)
    plt.close()
    print(f'保存 08_radar_overall.png')

    print(f'\n所有图保存至: {out_dir}')


if __name__ == '__main__':
    main()
