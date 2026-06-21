"""Clean figures (minimal style): ONLY algorithm names (legend / x-ticks),
axis labels, and numeric ticks/values. No titles, no footnotes, no extra text.

Outputs 9 figures to docs/figures_clean/:
  comparison  : cmp_delay_pdf, cmp_energy_pdf, cmp_satisfaction,
                cmp_cumulative_hl, cmp_queue_backlog
  ablation    : abl_delay_pdf, abl_energy_pdf, abl_satisfaction,
                abl_cumulative_hl
Reads docs/series_lh4_n25.json. System totals at N=192 (delay/energy x 192/25;
HL/queue per-sat x 192). Run: python plot_clean_figures.py
"""
import json, os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S    = json.load(open(os.path.join(os.path.dirname(__file__), 'docs', 'series_lh4_n25.json')))
OUT  = os.path.join(os.path.dirname(__file__), 'docs', 'figures_clean')
os.makedirs(OUT, exist_ok=True)
RATIO = 192 / 25
NSAT  = 192

COMP = [p for p in ['LyaMAPPO', 'TD3Sched', 'MHSPO', 'GDCO', 'LocalOnly'] if p in S]
ABL  = [p for p in ['LyaMAPPO', 'MAPPO_NoBat', 'LyapunovGreedy'] if p in S]
COLOR = {'LyaMAPPO':'#d62728', 'TD3Sched':'#1f77b4', 'MHSPO':'#2ca02c', 'GDCO':'#9467bd',
         'LocalOnly':'#7f7f7f', 'MAPPO_NoBat':'#e377c2', 'LyapunovGreedy':'#8c564b'}
MARK  = {'LyaMAPPO':'s', 'TD3Sched':'o', 'MHSPO':'^', 'GDCO':'P', 'LocalOnly':'*',
         'MAPPO_NoBat':'D', 'LyapunovGreedy':'v'}
DISP  = {'LocalOnly':'LSO', 'MAPPO_NoBat':'MAPPO-NoDOD'}
disp  = lambda p: DISP.get(p, p)
col   = lambda p: COLOR.get(p, '#333')
cum   = lambda a: np.cumsum(np.asarray(a, float))


def _save(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=160)
    plt.close()


def pdf_fig(values, order, xlabel, fname, smooth=2.2):
    allv = np.concatenate([np.asarray(values[p], float) for p in order])
    allv = allv[np.isfinite(allv)]
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in order:
        v = np.asarray(values[p], float); v = v[np.isfinite(v)]
        if len(v) < 10 or v.std() < 1e-9:
            continue
        kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * smooth)
        ys = kde(xs)
        big = (p == 'LyaMAPPO')
        ax.plot(xs, ys, color=col(p), lw=2.6 if big else 1.6, zorder=3 if big else 2)
        mk = np.linspace(0, len(xs) - 1, 18).astype(int)
        ax.plot(xs[mk], ys[mk], color=col(p), marker=MARK[p], ls='none',
                markersize=6 if big else 5, label=disp(p), zorder=3 if big else 2)
    ax.set_xlabel(xlabel); ax.set_ylabel('Probability density')
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.3)
    _save(fname)


def sat_bar(order, fname):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    vals = [S[p]['satisfaction'] for p in order]
    bars = ax.bar(range(len(order)), vals, color=[col(p) for p in order],
                  edgecolor='black', lw=0.6)
    bars[order.index('LyaMAPPO')].set_linewidth(2.4)
    ax.bar_label(bars, fmt='%.3f', fontsize=10)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([disp(p) for p in order], fontsize=10)
    ax.set_ylabel('User satisfaction'); ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', ls=':', alpha=0.3)
    _save(fname)


def cum_hl_fig(order, fname):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for p in order:
        y = cum(S[p]['hl']) * NSAT
        ax.plot(np.arange(len(y)), y, color=col(p), label=disp(p),
                lw=2.6 if p == 'LyaMAPPO' else 1.6, zorder=3 if p == 'LyaMAPPO' else 2)
    ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.3)
    _save(fname)


def queue_fig(order, fname, w=150):
    smooth = lambda a: (np.convolve(np.asarray(a, float), np.ones(w) / w, mode='valid')
                        if len(a) >= w else np.asarray(a, float))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for p in order:
        q = smooth(np.asarray(S[p]['queue_tasks'], float) * NSAT)
        ax.plot(np.arange(len(q)), q, color=col(p), label=disp(p),
                lw=2.6 if p == 'LyaMAPPO' else 1.4, zorder=3 if p == 'LyaMAPPO' else 2)
    ax.set_xlabel('Time slot'); ax.set_ylabel('System queue backlog (tasks)')
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.3)
    _save(fname)


# ── system totals at N=192 ──
delay  = {p: np.asarray(S[p]['delay'],  float) * RATIO       for p in S}        # s
energy = {p: np.asarray(S[p]['energy'], float) * RATIO / 1e3 for p in S}        # kJ

# comparison
pdf_fig(delay,  COMP, 'System total delay (s)',   'cmp_delay_pdf.png')
pdf_fig(energy, COMP, 'System total energy (kJ)', 'cmp_energy_pdf.png')
sat_bar(COMP, 'cmp_satisfaction.png')
cum_hl_fig(COMP, 'cmp_cumulative_hl.png')
queue_fig(COMP, 'cmp_queue_backlog.png')
# ablation
pdf_fig(delay,  ABL, 'System total delay (s)',   'abl_delay_pdf.png')
pdf_fig(energy, ABL, 'System total energy (kJ)', 'abl_energy_pdf.png')
sat_bar(ABL, 'abl_satisfaction.png')
cum_hl_fig(ABL, 'abl_cumulative_hl.png')

print('Clean figures written to', OUT)
for f in ['cmp_delay_pdf', 'abl_delay_pdf', 'cmp_energy_pdf', 'abl_energy_pdf',
          'cmp_satisfaction', 'abl_satisfaction', 'cmp_cumulative_hl',
          'abl_cumulative_hl', 'cmp_queue_backlog']:
    print(' -', f + '.png')
