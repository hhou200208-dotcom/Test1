"""Plot the 4 comparison figures (5 methods) from cached new-params series.
Reads docs/newparams_series/series.json (no eval). Consistent colors == the
cumulative-HL comparison figure. Figures:
  1 satisfaction PDF   2 delay PDF   3 energy PDF   4 cumulative HL (time series)
"""
import os, json
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'docs/newparams_series'
data = json.load(open(os.path.join(OUT, 'series.json')))
RATIO = 192.0 / 25; NSAT = 192
T = len(data['LyaMAPPO']['hl'])

ORDER = ['LyaMAPPO', 'MADDPG_DoD', 'MHSPO', 'GDCO', 'LocalOnly']
DISP = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD', 'LocalOnly': 'LSO'}
disp = lambda k: DISP.get(k, k)
COLOR = {'LyaMAPPO': '#d62728', 'MADDPG_DoD': '#F2C200', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}
MARK = {'LyaMAPPO': 's', 'MADDPG_DoD': 'D', 'MHSPO': '^', 'GDCO': 'P', 'LocalOnly': '*'}
big = lambda k: k == 'LyaMAPPO'
present = [k for k in ORDER if k in data]


def kde_fig(vals, xlabel, fname, smooth=2.2):
    allv = np.concatenate([np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for v in vals.values()])
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in present:
        v = np.asarray(vals[k], float); v = v[np.isfinite(v)]
        if len(v) < 10 or v.std() < 1e-9:
            continue
        kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * smooth); ys = kde(xs)
        ax.plot(xs, ys, color=COLOR[k], lw=2.4 if big(k) else 1.6, zorder=3 if big(k) else 2)
        mk = np.linspace(0, len(xs) - 1, 16).astype(int)
        ax.plot(xs[mk], ys[mk], color=COLOR[k], marker=MARK[k], ls='none',
                markersize=6 if big(k) else 5, label=disp(k), zorder=3 if big(k) else 2)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel); ax.set_ylabel('Probability density')
    ax.legend(fontsize=9, ncol=2); ax.grid(True, ls=':', alpha=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUT, f'{fname}.{ext}'), dpi=175)
    plt.close()


# 1 satisfaction PDF (rate, no projection)
sat = {}
for k in present:
    s = np.asarray(data[k]['sat_slot'], float); d = np.asarray(data[k]['sat_denom'], float); sat[k] = s[d > 0]
kde_fig(sat, 'User satisfaction', 'cmp_satisfaction_newparams')
# 2 delay PDF (x192/25)
kde_fig({k: np.asarray(data[k]['D']) * RATIO for k in present}, 'System total delay (s)', 'cmp_delay_newparams')
# 3 energy PDF (x192/25, kJ)
kde_fig({k: np.asarray(data[k]['E']) * RATIO / 1e3 for k in present}, 'System total energy', 'cmp_energy_newparams')

# 4 cumulative HL (time series, xNSAT)
fig, ax = plt.subplots(figsize=(7, 5))
for k in present:
    y = np.cumsum(np.asarray(data[k]['hl'], float)) * NSAT
    ax.plot(np.arange(T), y, color=COLOR[k], lw=2.4 if big(k) else 1.6, label=disp(k), zorder=3 if big(k) else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
ax.legend(fontsize=9, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'cmp_cumulative_hl_newparams.{ext}'), dpi=175)
plt.close()
print('present methods:', present, flush=True)
print('[done] 4 figures ->', OUT, flush=True)
