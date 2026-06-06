"""
evaluation/plotting.py
======================
所有绘图函数和曲线数据提取工具。

使用示例
--------
    from evaluation.plotting import plot_metric_per_run, plot_delay_pdf

    curves_by_run = [_collect_baseline_curves(run_recorders) for ...]
    plot_metric_per_run(curves_by_run, 'completion_rate', 'CR', 'Completion Rate',
                        output_dir='./figures', filename_prefix='cr')
    plot_delay_pdf(curves_by_run, output_dir='./figures')
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, TYPE_CHECKING

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from evaluation.recorder import MetricsRecorder

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

ALGORITHM_STYLES = {
    'MAPPO':          {'color': '#2196F3', 'linestyle': '-',  'marker': 'o', 'label': 'MAPPO (Proposed)'},
    'LocalOnly':      {'color': '#F44336', 'linestyle': '--', 'marker': 's', 'label': 'Local Only'},
    'GreedyDelay':    {'color': '#FF9800', 'linestyle': ':',  'marker': '^', 'label': 'Greedy Delay'},
    'MAPPO_NoDod':    {'color': '#9C27B0', 'linestyle': '-.', 'marker': 'D', 'label': 'MAPPO w/o DoD'},
    'LyapunovGreedy': {'color': '#4CAF50', 'linestyle': '--', 'marker': 'v', 'label': 'Lyapunov Greedy'},
    'MHSPO':          {'color': '#00BCD4', 'linestyle': '-.', 'marker': 'P',
                       'label': 'MHSPO (Zhang et al. TMC 2024)'},
}


def get_style(algorithm: str) -> Dict:
    return ALGORITHM_STYLES.get(
        algorithm,
        {'color': '#607D8B', 'linestyle': '-', 'marker': 'x', 'label': algorithm},
    )


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── 曲线数据提取 ──────────────────────────────────────────────
def _collect_baseline_curves(recorders: Dict[str, "MetricsRecorder"],
                              phase: str = 'eval',
                              delay_sample_interval: int = 5) -> Dict:
    """从 recorder 中提取各指标时间序列，用于 plot_metric_per_run。"""
    result = {}
    for alg, rec in recorders.items():
        records = [r for r in rec._slot_records if r['phase'] == phase]
        delay_samples: List[float] = []
        for i, r in enumerate(records):
            if i % delay_sample_interval == 0:
                delay_samples.extend(r.get('_slot_e2e_delays', []))
        result[alg] = {
            'slots':               [r['slot'] for r in records],
            'completion_rate':     [r['completion_rate'] for r in records],
            'avg_dod':             [r['avg_dod'] for r in records],
            'health_loss':         list(np.cumsum([r['avg_health_loss'] for r in records])),
            'total_queue':         [r['total_queue_size'] / 1e6 for r in records],
            'forwarded':           [r['forwarded'] for r in records],
            'slot_satisfaction_rate': [r['slot_satisfaction_rate'] for r in records],
            'satisfaction_samples_a': [r['slot_satisfaction_rate'] for r in records],
            'satisfaction_samples_b': [r.get('slot_satisfaction_rate_orig', 0.0) for r in records],
            'delay_samples':       delay_samples,
        }
    return result


def _get_dod_snapshots(recorders: Dict[str, "MetricsRecorder"],
                       snapshot_interval: int = 360,
                       phase: str = 'eval') -> Dict:
    """提取每隔 snapshot_interval 时隙的各卫星 DoD 快照。"""
    snapshots = {alg: {} for alg in recorders}
    for alg, rec in recorders.items():
        records = [r for r in rec._slot_records if r['phase'] == phase]
        for i, r in enumerate(records):
            if (i + 1) % snapshot_interval == 0:
                snap_idx = (i + 1) // snapshot_interval
                dods = r.get('_per_sat_dod', [])
                if dods:
                    snapshots[alg][snap_idx] = dods
    return snapshots


# ── 多run图 ───────────────────────────────────────────────────
def plot_metric_per_run(curves_by_run: List[Dict], metric: str,
                        ylabel: str, title: str,
                        output_dir: str, filename_prefix: str) -> None:
    """每个指标绘制 N_run 张单图 + 1张均值汇总图。"""
    os.makedirs(output_dir, exist_ok=True)
    alg_names = list(curves_by_run[0].keys())
    n_runs    = len(curves_by_run)

    for run_idx, curves in enumerate(curves_by_run):
        fig, ax = plt.subplots(figsize=(10, 5))
        for alg in alg_names:
            style = get_style(alg)
            data  = curves[alg]
            slots, vals = data['slots'], data[metric]
            step = max(1, len(slots) // 20)
            ax.plot(slots, vals, color=style['color'], linestyle=style['linestyle'],
                    label=style['label'], linewidth=1.5,
                    marker=style['marker'], markevery=step, markersize=4)
        ax.set_xlabel('Time Slot'); ax.set_ylabel(ylabel)
        ax.set_title(f'{title} — Run {run_idx}')
        ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_prefix}_run{run_idx}.png'))
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for alg in alg_names:
        style   = get_style(alg)
        min_len = min(len(c[alg][metric]) for c in curves_by_run)
        mat     = np.array([c[alg][metric][:min_len] for c in curves_by_run])
        slots   = curves_by_run[0][alg]['slots'][:min_len]
        mean, ci = mat.mean(axis=0), 1.96 * mat.std(axis=0) / np.sqrt(n_runs)
        step    = max(1, len(slots) // 20)
        ax.plot(slots, mean, color=style['color'], linestyle=style['linestyle'],
                label=style['label'], linewidth=1.5,
                marker=style['marker'], markevery=step, markersize=4)
        ax.fill_between(slots, mean - ci, mean + ci, alpha=0.15, color=style['color'])
    ax.set_xlabel('Time Slot'); ax.set_ylabel(ylabel)
    ax.set_title(f'{title} — Mean ± 95% CI ({n_runs} runs)')
    ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_mean.png'))
    plt.close()


# ── 比较图 ────────────────────────────────────────────────────
def plot_dod_comparison(results: Dict[str, str], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg_name, result_dir in results.items():
        curves = _load_json(os.path.join(result_dir, 'curves.json'))
        if not curves or not curves.get('dod_curve'):
            continue
        dc    = curves['dod_curve']
        style = get_style(alg_name)
        ax.plot(dc['slots'], dc['avg_dod'], color=style['color'],
                linestyle=style['linestyle'], label=style['label'], linewidth=1.5)
        ax.fill_between(dc['slots'],
                        np.array(dc['avg_dod']) - np.array(dc['std_dod']),
                        np.array(dc['avg_dod']) + np.array(dc['std_dod']),
                        alpha=0.15, color=style['color'])
    ax.set_xlabel('Time Slot'); ax.set_ylabel('Average DoD')
    ax.set_title('Battery DoD Comparison')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dod_comparison.png')); plt.close()


def plot_health_loss_comparison(results: Dict[str, str], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg_name, result_dir in results.items():
        curves = _load_json(os.path.join(result_dir, 'curves.json'))
        if not curves or not curves.get('health_loss_curve'):
            continue
        hlc   = curves['health_loss_curve']
        style = get_style(alg_name)
        ax.plot(hlc['slots'], hlc['cumulative_health_loss'],
                color=style['color'], linestyle=style['linestyle'],
                label=style['label'], linewidth=1.5)
    ax.set_xlabel('Time Slot'); ax.set_ylabel('Cumulative Health Loss')
    ax.set_title('Battery Health Loss Comparison')
    ax.legend(); ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'health_loss_comparison.png')); plt.close()


def plot_queue_comparison(results: Dict[str, str], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg_name, result_dir in results.items():
        curves = _load_json(os.path.join(result_dir, 'curves.json'))
        if not curves or not curves.get('queue_curve'):
            continue
        qc    = curves['queue_curve']
        style = get_style(alg_name)
        ax.plot(qc['slots'], np.array(qc['avg_qf_size']) / 1e6,
                color=style['color'], linestyle=style['linestyle'],
                label=style['label'], linewidth=1.5)
    ax.set_xlabel('Time Slot'); ax.set_ylabel('Avg Forward Queue Size (MB)')
    ax.set_title('Queue Backlog Comparison')
    ax.legend(); ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'queue_comparison.png')); plt.close()


def plot_completion_rate_bar(summaries: Dict, output_dir: str) -> None:
    alg_names = list(summaries.keys())
    means  = [summaries[a]['completion_rate']['mean'] for a in alg_names]
    cis    = [summaries[a]['completion_rate']['ci95']  for a in alg_names]
    colors = [get_style(a)['color'] for a in alg_names]
    fig, ax = plt.subplots(figsize=(10, 6))
    x    = np.arange(len(alg_names))
    bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([get_style(a)['label'] for a in alg_names], rotation=15, ha='right')
    ax.set_ylabel('Task Completion Rate')
    ax.set_title('Completion Rate Comparison (95% CI)')
    ax.set_ylim([0, 1.1]); ax.grid(True, axis='y', alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'completion_rate_bar.png')); plt.close()


def generate_comparison_table(summaries: Dict, output_dir: str) -> None:
    import csv as _csv
    os.makedirs(output_dir, exist_ok=True)
    metrics = [('completion_rate', 'Completion Rate'), ('avg_dod', 'Avg DoD'),
               ('avg_health_loss', 'Health Loss'), ('avg_qf_size', 'Avg QF Size (MB)')]
    rows = []
    for alg_name, summary in summaries.items():
        row = {'Algorithm': get_style(alg_name)['label']}
        for mk, mn in metrics:
            if mk in summary:
                m = summary[mk]
                row[mn] = (f"{m['mean']/1e6:.2f} ± {m['ci95']/1e6:.2f}"
                           if mk == 'avg_qf_size'
                           else f"{m['mean']:.4f} ± {m['ci95']:.4f}")
            else:
                row[mn] = 'N/A'
        rows.append(row)
    if rows:
        with open(os.path.join(output_dir, 'comparison_table.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)


# ── DoD 快照直方图 ────────────────────────────────────────────
def plot_dod_snapshots(snapshots_by_run: List[Dict], output_dir: str,
                       n_sats: int = 25, n_snapshots: int = 10,
                       snapshot_interval: int = 360) -> None:
    os.makedirs(output_dir, exist_ok=True)
    alg_names   = list(snapshots_by_run[0].keys())
    bins        = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_width   = 0.8 / len(alg_names)
    for snap_idx in range(1, n_snapshots + 1):
        fig, ax = plt.subplots(figsize=(10, 5))
        for alg_i, alg in enumerate(alg_names):
            style    = get_style(alg)
            all_dods = []
            for run_snaps in snapshots_by_run:
                all_dods.extend(run_snaps.get(alg, {}).get(snap_idx, []))
            if not all_dods:
                continue
            counts, _ = np.histogram(all_dods, bins=bins)
            counts    = counts / len(snapshots_by_run)
            offset    = (alg_i - len(alg_names) / 2 + 0.5) * bar_width
            ax.bar(bin_centers + offset, counts, width=bar_width * 0.9,
                   color=style['color'], alpha=0.8, label=style['label'],
                   edgecolor='white', linewidth=0.5)
        slot_num = snap_idx * snapshot_interval
        ax.set_xlabel('DoD Value'); ax.set_ylabel('Number of Satellites (avg over runs)')
        ax.set_title(f'Per-Satellite DoD Distribution — Slot {slot_num}')
        ax.set_xlim(0, 1); ax.legend(loc='upper right', fontsize=9); ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dod_snapshot_slot{slot_num:04d}.png'))
        plt.close()


# ── 时延 PDF ──────────────────────────────────────────────────
def plot_delay_pdf(curves_by_run: List[Dict], output_dir: str,
                   filename: str = 'delay_pdf',
                   delay_sample_interval: int = 5) -> None:
    from scipy.stats import gaussian_kde
    os.makedirs(output_dir, exist_ok=True)
    alg_names = list(curves_by_run[0].keys())
    samples_dict: Dict[str, np.ndarray] = {}
    for alg in alg_names:
        raw = []
        for rc in curves_by_run:
            raw.extend(rc[alg].get('delay_samples', []))
        arr = np.array(raw, dtype=np.float64)
        samples_dict[alg] = arr[arr > 0]

    all_vals = np.concatenate([v for v in samples_dict.values() if len(v) > 0])
    if len(all_vals) == 0:
        print('[plot_delay_pdf] 无有效样本，跳过'); return
    x_min  = max(all_vals.min() - 0.5, 0.0)
    x_max  = all_vals.max() + 0.5
    x_grid = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(9, 5))
    for alg in alg_names:
        arr   = samples_dict[alg]
        style = get_style(alg)
        if len(arr) < 10:
            continue
        kde = gaussian_kde(arr, bw_method='scott')
        ax.plot(x_grid, kde(x_grid), color=style['color'], linestyle=style['linestyle'],
                linewidth=2.0, marker=style['marker'],
                markevery=max(1, len(x_grid) // 12), markersize=5,
                label=f"{style['label']} (μ={arr.mean():.2f}s, n={len(arr)})")
    ax.set_xlabel('End-to-End Delay (s)'); ax.set_ylabel('Probability Density')
    ax.set_title(f'Distribution of Task E2E Delay\n'
                 f'(sampled every {delay_sample_interval} slots)')
    ax.set_xlim(x_min, x_max); ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300); plt.close()


# ── 满意度 PDF ────────────────────────────────────────────────
def plot_satisfaction_pdf(curves_by_run: List[Dict], output_dir: str,
                          sample_key: str = 'satisfaction_samples_a',
                          filename: str = 'satisfaction_pdf') -> None:
    from scipy.stats import gaussian_kde
    os.makedirs(output_dir, exist_ok=True)
    alg_names = list(curves_by_run[0].keys())
    samples_dict: Dict[str, np.ndarray] = {}
    for alg in alg_names:
        raw = []
        for rc in curves_by_run:
            raw.extend(rc[alg].get(sample_key, []))
        arr = np.array(raw, dtype=np.float64)
        samples_dict[alg] = arr[(arr >= 0.0) & (arr <= 1.0 + 1e-9)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in alg_names:
        arr   = samples_dict[alg]
        style = get_style(alg)
        if len(arr) < 10:
            continue
        if arr.std() < 1e-9:
            ax.axvline(arr[0], color=style['color'], linestyle=style['linestyle'],
                       linewidth=2.0, label=f"{style['label']} (μ={arr.mean():.3f}, σ≈0)")
            continue
        kde = gaussian_kde(arr, bw_method='scott')
        x   = np.linspace(max(arr.min() - 0.05, 0.0), min(arr.max() + 0.05, 1.0), 500)
        ax.plot(x, kde(x), color=style['color'], linestyle=style['linestyle'],
                linewidth=2.0, marker=style['marker'],
                markevery=max(1, len(x) // 12), markersize=5,
                label=f"{style['label']} (μ={arr.mean():.3f})")
    ax.set_xlabel('Satisfaction'); ax.set_ylabel('Distribution')
    ax.set_title("Comparison of users' satisfaction distribution")
    ax.set_xlim(0.0, 1.05); ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300); plt.close()
