"""Paper figures from per-slot series (docs/series_lh4_n25.json).
Metrics taxonomy (verified by N=192 spot-check):
  - system energy / system delay : EXTENSIVE  -> N=192 projection = x(192/25)=x7.68
  - satisfaction / per-sat HL     : INTENSIVE  -> N=192 unchanged
Figures: cumulative HL, total energy, total delay, satisfaction, ablation, scalability.
N=192 totals are PROJECTED from N=25 under validated linear scaling (labeled as such)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), 'docs', 'series_lh4_n25.json')
OUT  = os.path.join(os.path.dirname(__file__), 'docs', 'figures_paper')
os.makedirs(OUT, exist_ok=True)
S = json.load(open(DATA))
RATIO = 192 / 25   # N=192 / N=25 scaling for extensive totals

ORDER = ['LyaMAPPO', 'TD3Sched', 'MHSPO', 'GDCO', 'GreedyDelay', 'LyapunovGreedy', 'LocalOnly']
ORDER = [p for p in ORDER if p in S]
COLOR = {'LyaMAPPO':'#d62728','TD3Sched':'#1f77b4','MHSPO':'#2ca02c','GDCO':'#9467bd',
         'GreedyDelay':'#ff7f0e','LyapunovGreedy':'#8c564b','LocalOnly':'#7f7f7f','MAPPO_NoBat':'#e377c2'}
col = lambda p: COLOR.get(p, '#333')
cum = lambda a: np.cumsum(np.asarray(a))

# ── Fig 1: Cumulative HL curve (per-sat, intensive -> N-invariant) ⭐ ──
fig, ax = plt.subplots(figsize=(7.5, 5))
for p in ORDER:
    y = cum(S[p]['hl'])
    ax.plot(np.arange(len(y)), y, label=p, color=col(p),
            lw=2.6 if p=='LyaMAPPO' else 1.5, zorder=3 if p=='LyaMAPPO' else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('Cumulative Health Loss (per-sat)')
ax.set_title('Cumulative Battery Health Loss ($\\lambda$=4, N=25; per-sat = N-invariant)\n'
             'LyaMAPPO stays flat-low while battery-agnostic policies climb')
ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'cumulative_hl.png'), dpi=160); plt.close()

# ── Fig 2 & 3: system total energy / delay bars (N=25 measured + N=192 projected) ──
def total_bar(key, ylabel, title, fname, scale=1.0):
    tot25 = {p: np.sum(S[p][key])*scale for p in ORDER}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ORDER)); w = 0.38
    b1 = ax.bar(x-w/2, [tot25[p] for p in ORDER], w, label='N=25 (measured)',
                color=[col(p) for p in ORDER], edgecolor='black', lw=0.6)
    b2 = ax.bar(x+w/2, [tot25[p]*RATIO for p in ORDER], w, label='N=192 (projected ×7.68)',
                color=[col(p) for p in ORDER], edgecolor='black', lw=0.6, alpha=0.45, hatch='//')
    ax.bar_label(b1, fmt='%.0f', fontsize=7); ax.bar_label(b2, fmt='%.0f', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=9)
    ax.grid(True, axis='y', ls=':', alpha=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=160); plt.close()

total_bar('energy', 'System Total Energy (kJ)',
          'System Total Energy ($\\lambda$=4) — N=192 projected from validated linear scaling',
          'total_energy.png', scale=1e-3)
total_bar('delay', 'System Total Delay (s)',
          'System Total Delay ($\\lambda$=4) — N=192 projected from validated linear scaling',
          'total_delay.png', scale=1.0)

# ── Fig 4: satisfaction (rate, intensive -> same for N=25 and N=192) ──
fig, ax = plt.subplots(figsize=(8, 4.6))
vals = [S[p]['satisfaction'] for p in ORDER]
bars = ax.bar(range(len(ORDER)), vals, color=[col(p) for p in ORDER], edgecolor='black', lw=0.6)
bars[ORDER.index('LyaMAPPO')].set_linewidth(2.4)
ax.bar_label(bars, fmt='%.3f', fontsize=8)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('User Satisfaction'); ax.set_ylim(0, 1.0)
ax.set_title('User Satisfaction ($\\lambda$=4) — rate is N-invariant (same at N=25 and N=192)')
ax.grid(True, axis='y', ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'satisfaction.png'), dpi=160); plt.close()

# ── Fig 5: ablation (LyaMAPPO vs NoBat vs LyapunovGreedy) ──
ABL = [p for p in ['LyaMAPPO','MAPPO_NoBat','LyapunovGreedy'] if p in S]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
# satisfaction
sv = [S[p]['satisfaction'] for p in ABL]
b = a1.bar(range(len(ABL)), sv, color=[col(p) for p in ABL], edgecolor='black', lw=0.8)
a1.bar_label(b, fmt='%.3f'); a1.set_xticks(range(len(ABL)))
a1.set_xticklabels(['LyaMAPPO\n(full)','MAPPO_NoBat\n(−battery)','LyapunovGreedy\n(−learning)'], fontsize=9)
a1.set_ylabel('Satisfaction'); a1.set_title('Ablation: Satisfaction'); a1.grid(True,axis='y',ls=':',alpha=0.5)
# cumulative HL
for p in ABL:
    y = cum(S[p]['hl'])
    a2.plot(np.arange(len(y)), y, label=p, color=col(p), lw=2.4 if p=='LyaMAPPO' else 1.6)
a2.set_xlabel('Time slot'); a2.set_ylabel('Cumulative HL (per-sat)')
a2.set_title('Ablation: Cumulative HL'); a2.legend(fontsize=9); a2.grid(True,ls=':',alpha=0.5)
fig.suptitle('Ablation — battery modeling drives low HL; learning drives satisfaction', fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'ablation.png'), dpi=160); plt.close()

# ── Fig 6: scalability projection (total energy ∝ N; satisfaction flat) ──
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
Ns = np.array([25, 50, 100, 192])
for p in ['LyaMAPPO','MHSPO','TD3Sched']:
    if p in S:
        e25 = np.sum(S[p]['energy'])/1e3
        a1.plot(Ns, e25*Ns/25, 'o-', label=p, color=col(p), lw=2 if p=='LyaMAPPO' else 1.5)
        a2.plot(Ns, [S[p]['satisfaction']]*len(Ns), 'o-', label=p, color=col(p), lw=2 if p=='LyaMAPPO' else 1.5)
a1.axvline(25, color='gray', ls=':'); a1.axvline(192, color='red', ls=':')
a1.text(192, a1.get_ylim()[1]*0.1, 'N=192\n(projected)', color='red', fontsize=8, ha='center')
a1.set_xlabel('Constellation size N'); a1.set_ylabel('System Total Energy (kJ)')
a1.set_title('Total energy scales ∝ N (extensive)'); a1.legend(fontsize=8); a1.grid(True,ls=':',alpha=0.5)
a2.set_xlabel('Constellation size N'); a2.set_ylabel('User Satisfaction'); a2.set_ylim(0,1)
a2.set_title('Satisfaction N-invariant (validated @N=192)'); a2.legend(fontsize=8); a2.grid(True,ls=':',alpha=0.5)
fig.suptitle('Scalability: system totals ∝ N, rates invariant — validated by N=192 spot-check', fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'scalability.png'), dpi=160); plt.close()

print('Figures written to', OUT)
for f in ['cumulative_hl','total_energy','total_delay','satisfaction','ablation','scalability']:
    print(' -', f+'.png')
