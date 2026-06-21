"""Paper figures (satisfaction metric, no DoD result chart):
   (1) Pareto satisfaction-vs-HL, (2) 4-metric bar chart.
Reads /tmp/scoreboard7.json (from eval_all7_scalars.py), writes docs/figures_paper/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), 'docs', 'scoreboard7_lh4.json')
OUT  = os.path.join(os.path.dirname(__file__), 'docs', 'figures_paper')
os.makedirs(OUT, exist_ok=True)
res = json.load(open(DATA))

ORDER = ['LyaMAPPO', 'TD3Sched', 'MHSPO', 'GDCO', 'GreedyDelay', 'LyapunovGreedy', 'LocalOnly']
ORDER = [p for p in ORDER if p in res]
COLOR = {'LyaMAPPO':'#d62728', 'TD3Sched':'#1f77b4', 'MHSPO':'#2ca02c', 'GDCO':'#9467bd',
         'GreedyDelay':'#ff7f0e', 'LyapunovGreedy':'#8c564b', 'LocalOnly':'#7f7f7f'}
col = lambda p: COLOR.get(p, '#333333')
hl4 = lambda p: res[p]['hl'] * 1e4   # HL in units of 1e-4

# ── Fig 1: Pareto  Satisfaction (x) vs HL (y) ───────────────
fig, ax = plt.subplots(figsize=(7, 5.2))
ymax = max(hl4(p) for p in ORDER) * 1.15
ax.axhspan(0, 2.0, color='#d62728', alpha=0.06, zorder=0)
ax.text(0.30, 0.9, 'Low-HL region\n(battery-friendly)', fontsize=9, color='#d62728', alpha=0.85)
for p in ORDER:
    x, y = res[p]['satisfaction'], hl4(p)
    big = (p == 'LyaMAPPO')
    ax.scatter(x, y, s=340 if big else 170, c=col(p), edgecolors='black',
               linewidths=1.6 if big else 0.8, zorder=3, marker='*' if big else 'o')
    ax.annotate(p, (x, y), textcoords='offset points',
                xytext=(9, 9 if not big else -18), fontsize=10,
                fontweight='bold' if big else 'normal', color=col(p))
ax.annotate('', xy=(0.86, ymax*0.12), xytext=(0.60, ymax*0.12),
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.text(0.66, ymax*0.16, 'better satisfaction →', fontsize=9, color='gray')
ax.text(0.255, ymax*0.55, 'lower HL\n(better) ↓', fontsize=9, color='gray')
ax.set_xlabel('Satisfaction  (satisfied / (done + timeout))', fontsize=11)
ax.set_ylabel('Battery Health Loss  HL  (×10$^{-4}$ / slot)', fontsize=11)
ax.set_title('Pareto: Satisfaction vs Battery Health Loss (N=192, $\\lambda$=4)\n'
             'LyaMAPPO uniquely occupies the low-HL frontier (rate/per-sat metrics N-invariant)', fontsize=12)
ax.set_ylim(0, ymax); ax.set_xlim(0.20, 0.92)
ax.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'pareto_sat_hl.png'), dpi=160); plt.close()

# ── Fig 2: 4-metric bar chart (2x2) ─────────────────────────
metrics = [('satisfaction', 'Satisfaction (higher better)',      1.0, '{:.3f}'),
           ('delay',        'E2E Delay (s, lower better)',        1.0, '{:.2f}'),
           ('hl',           'Battery Health Loss (×10$^{-4}$, lower better)', 1e4, '{:.2f}'),
           ('queue_tasks',  'Queue Backlog (tasks/sat, lower better)', 1.0, '{:.1f}')]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, title, scale, fmt) in zip(axes.flat, metrics):
    vals = [res[p][key] * scale for p in ORDER]
    bars = ax.bar(range(len(ORDER)), vals, color=[col(p) for p in ORDER],
                  edgecolor='black', linewidth=0.6)
    bars[ORDER.index('LyaMAPPO')].set_linewidth(2.4)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis='y', ls=':', alpha=0.5)
fig.suptitle('7-Policy Comparison (N=192, $\\lambda$=4, 5400 slots) — LyaMAPPO dominates on HL', fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'bars_4metrics.png'), dpi=160); plt.close()

print('Figures written:')
print(' -', os.path.join(OUT, 'pareto_sat_hl.png'))
print(' -', os.path.join(OUT, 'bars_4metrics.png'))
