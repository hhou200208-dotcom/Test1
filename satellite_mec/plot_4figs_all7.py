"""Plot 4 comparison figures for 7 methods from a per-slot series JSON.

Self-contained: reads series_all7.json (same directory) and produces
  1. satisfaction PDF   2. system delay PDF
  3. system energy PDF  4. system cumulative health loss (time series)

Data schema (series_all7.json): {method: {E, D, hl, sat_slot, sat_denom}}, each a
per-slot list over T=5400 slots (1 eval seed). Extensive quantities are projected
from the 25-satellite simulation to the 192-satellite system by x(192/25);
satisfaction is an intensive rate (no projection). Scenario: LEO Walker,
lambda_high=4, kappa=1.5e-27, E_cap=54 kJ, V_f=2e17, DoD in [0,0.8].

Usage:  python plot_4figs_all7.py
Requires: numpy, scipy, matplotlib.
"""
import os
import json
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'series_all7.json')
OUT = HERE
RATIO = 192.0 / 25.0     # 25-sat sim -> 192-sat system projection (extensive quantities)
NSAT = 192               # system size for cumulative health loss

data = json.load(open(DATA))

# draw order (BLA-MAPPO emphasized first) and styling
ORDER = ['BLA-MAPPO', 'w/o DoD', 'w/o Task Prior', 'MHSPO', 'LyDRL-DoD', 'GDCO', 'LSO']
ORDER = [m for m in ORDER if m in data]
COLOR = {'BLA-MAPPO': '#d62728', 'w/o DoD': '#e377c2', 'w/o Task Prior': '#8c564b',
         'MHSPO': '#2ca02c', 'LyDRL-DoD': '#F2C200', 'GDCO': '#9467bd', 'LSO': '#7f7f7f'}
MARK = {'BLA-MAPPO': 's', 'w/o DoD': 'o', 'w/o Task Prior': 'P',
        'MHSPO': '^', 'LyDRL-DoD': 'D', 'GDCO': 'X', 'LSO': '*'}
big = lambda k: k == 'BLA-MAPPO'
T = len(data[ORDER[0]]['hl'])


def kde_fig(vals, xlabel, fname, smooth=2.2):
    """KDE probability-density plot of a per-method value dict."""
    allv = np.concatenate([np.asarray(v, float)[np.isfinite(np.asarray(v, float))]
                           for v in vals.values()])
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for k in ORDER:
        v = np.asarray(vals[k], float); v = v[np.isfinite(v)]
        if len(v) < 10 or v.std() < 1e-9:
            continue
        kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * smooth); ys = kde(xs)
        ax.plot(xs, ys, color=COLOR[k], lw=2.6 if big(k) else 1.6, zorder=3 if big(k) else 2)
        mk = np.linspace(0, len(xs) - 1, 16).astype(int)
        ax.plot(xs[mk], ys[mk], color=COLOR[k], marker=MARK[k], ls='none',
                markersize=6 if big(k) else 5, label=k, zorder=3 if big(k) else 2)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel); ax.set_ylabel('Probability density')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUT, f'{fname}.{ext}'), dpi=180)
    plt.close()


# 1. user satisfaction PDF (intensive rate; count only slots with departures)
sat = {}
for k in ORDER:
    s = np.asarray(data[k]['sat_slot'], float)
    d = np.asarray(data[k]['sat_denom'], float)
    sat[k] = s[d > 0]
kde_fig(sat, 'User satisfaction', 'fig1_satisfaction_pdf')

# 2. system total delay PDF (extensive -> x192/25)
kde_fig({k: np.asarray(data[k]['D']) * RATIO for k in ORDER},
        'System total delay (s)', 'fig2_delay_pdf')

# 3. system total energy PDF (extensive -> x192/25, shown in kJ)
kde_fig({k: np.asarray(data[k]['E']) * RATIO / 1e3 for k in ORDER},
        'System total energy', 'fig3_energy_pdf')

# 4. system cumulative health loss over time (x192)
fig, ax = plt.subplots(figsize=(7.2, 5))
for k in ORDER:
    y = np.cumsum(np.asarray(data[k]['hl'], float)) * NSAT
    ax.plot(np.arange(T), y, color=COLOR[k], lw=2.6 if big(k) else 1.6, label=k,
            zorder=3 if big(k) else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'fig4_cumulative_hl.{ext}'), dpi=180)
plt.close()

print('[done] wrote fig1..fig4 (png+pdf) to', OUT)
