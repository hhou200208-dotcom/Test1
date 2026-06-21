"""PDF / distribution figures (reference Fig 5/6/7 style):
system delay overhead, system energy overhead, user satisfaction — per-slot
distributions over the 5400-slot evaluation, one KDE curve per policy.
Reads docs/series_lh4_n25.json. Energy/delay are system totals (extensive) ->
N=192 = x(192/25); satisfaction is a rate (N-invariant)."""
import json, os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S    = json.load(open(os.path.join(os.path.dirname(__file__), 'docs', 'series_lh4_n25.json')))
OUT  = os.path.join(os.path.dirname(__file__), 'docs', 'figures_paper')
RATIO = 192 / 25

ORDER = ['LyaMAPPO', 'TD3Sched', 'MHSPO', 'GDCO', 'GreedyDelay', 'LyapunovGreedy', 'LocalOnly']
ORDER = [p for p in ORDER if p in S]
COLOR = {'LyaMAPPO':'#d62728','TD3Sched':'#1f77b4','MHSPO':'#2ca02c','GDCO':'#9467bd',
         'GreedyDelay':'#ff7f0e','LyapunovGreedy':'#8c564b','LocalOnly':'#7f7f7f'}
MARK  = {'LyaMAPPO':'s','TD3Sched':'o','MHSPO':'^','GDCO':'P','GreedyDelay':'D',
         'LyapunovGreedy':'v','LocalOnly':'*'}

def kde_fig(vals, xlabel, title, fname, note=None, smooth=2.2):
    allv = np.concatenate([np.asarray(v, float) for v in vals.values()])
    allv = allv[np.isfinite(allv)]
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in ORDER:
        v = np.asarray(vals[p], float); v = v[np.isfinite(v)]
        if len(v) < 10 or v.std() < 1e-9:
            continue
        kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * smooth)   # 加大带宽 → 更平滑
        ys = kde(xs)
        big = (p == 'LyaMAPPO')
        ax.plot(xs, ys, color=COLOR[p], lw=2.6 if big else 1.6, zorder=3 if big else 2)
        mk = np.linspace(0, len(xs)-1, 18).astype(int)
        ax.plot(xs[mk], ys[mk], color=COLOR[p], marker=MARK[p], ls='none',
                markersize=6 if big else 5, label=p, zorder=3 if big else 2)
    ax.set_xlabel(xlabel); ax.set_ylabel('Distribution'); ax.set_title(title)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)
    if note:
        ax.text(0.5, -0.16, note, transform=ax.transAxes, ha='center',
                fontsize=7, color='gray', style='italic')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=160); plt.close()

PROJ = 'System totals reported at N=192 (projected from validated linear scaling).'

# ── system delay overhead distribution (per-slot system total delay, N=192) ──
delay = {p: np.asarray(S[p]['delay']) * RATIO for p in ORDER}
kde_fig(delay, 'System delay overhead (s)',
        'Distribution of System Delay Overhead (N=192, $\\lambda$=4)', 'delay_pdf.png', PROJ)

# ── system energy overhead distribution (per-slot system energy, N=192, kJ) ──
energy = {p: np.asarray(S[p]['energy']) * RATIO / 1e3 for p in ORDER}
kde_fig(energy, 'System energy overhead (kJ)',
        'Distribution of System Energy Overhead (N=192, $\\lambda$=4)', 'energy_pdf.png', PROJ)

# ── user satisfaction distribution (per-slot rate, active slots only; N-invariant) ──
sat = {}
for p in ORDER:
    s = np.asarray(S[p]['sat_slot'], float); d = np.asarray(S[p]['sat_denom'], float)
    sat[p] = s[d > 0]
kde_fig(sat, 'Satisfaction',
        'Distribution of User Satisfaction (N=192, $\\lambda$=4)', 'satisfaction_pdf.png')

print('PDF figures written:')
for f in ['delay_pdf', 'energy_pdf', 'satisfaction_pdf']:
    print(' -', f+'.png')
