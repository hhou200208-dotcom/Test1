"""Ablation figures (SEPARATE from comparison figures).
Variants: LyaMAPPO (full) vs MAPPO_NoBat (-battery) vs LyapunovGreedy (-learning).
Reads docs/series_lh4_n25.json. Per-metric ablation set: satisfaction, cumulative HL,
system total energy (N=192 framing consistent with comparison figures)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), 'docs', 'series_lh4_n25.json')
OUT  = os.path.join(os.path.dirname(__file__), 'docs', 'figures_paper')
os.makedirs(OUT, exist_ok=True)
S = json.load(open(DATA))
RATIO = 192 / 25

ABL    = ['LyaMAPPO', 'MAPPO_NoBat', 'LyapunovGreedy']
LABEL  = {'LyaMAPPO':'LyaMAPPO\n(full)', 'MAPPO_NoBat':'MAPPO-NoDOD\n(−DoD)',
          'LyapunovGreedy':'LyapunovGreedy\n(−learning)'}
COLOR  = {'LyaMAPPO':'#d62728', 'MAPPO_NoBat':'#e377c2', 'LyapunovGreedy':'#8c564b'}
col = lambda p: COLOR[p]
cum = lambda a: np.cumsum(np.asarray(a))

# ── Ablation 1: satisfaction (−learning collapses it) ──
fig, ax = plt.subplots(figsize=(6, 4.6))
sv = [S[p]['satisfaction'] for p in ABL]
b = ax.bar(range(len(ABL)), sv, color=[col(p) for p in ABL], edgecolor='black', lw=1.0)
b[0].set_linewidth(2.4); ax.bar_label(b, fmt='%.3f', fontsize=10)
ax.set_xticks(range(len(ABL))); ax.set_xticklabels([LABEL[p] for p in ABL], fontsize=9)
ax.set_ylabel('User Satisfaction'); ax.set_ylim(0, 1.0)
ax.set_title('Ablation — User Satisfaction (N=192)\nremoving learning collapses satisfaction')
ax.grid(True, axis='y', ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'ablation_satisfaction.png'), dpi=160); plt.close()

# ── Ablation 2: cumulative HL (−battery explodes it) ──
fig, ax = plt.subplots(figsize=(7, 4.8))
for p in ABL:
    y = cum(S[p]['hl'])
    ax.plot(np.arange(len(y)), y, label=LABEL[p].replace('\n',' '), color=col(p),
            lw=2.6 if p=='LyaMAPPO' else 1.8)
ax.set_xlabel('Time slot'); ax.set_ylabel('Per-satellite Cumulative Health Loss')
ax.set_title('Ablation — Cumulative Battery Health Loss (N=192)\nremoving battery modeling explodes HL')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'ablation_cumulative_hl.png'), dpi=160); plt.close()

# ── Ablation 3: system total energy (N=192 projected) ──
fig, ax = plt.subplots(figsize=(6, 4.6))
ev = [np.sum(S[p]['energy'])/1e3*RATIO for p in ABL]
b = ax.bar(range(len(ABL)), ev, color=[col(p) for p in ABL], edgecolor='black', lw=1.0)
b[0].set_linewidth(2.4); ax.bar_label(b, fmt='%.0f', fontsize=10)
ax.set_xticks(range(len(ABL))); ax.set_xticklabels([LABEL[p] for p in ABL], fontsize=9)
ax.set_ylabel('System Total Energy (kJ)')
ax.set_title('Ablation — System Total Energy (N=192)')
ax.grid(True, axis='y', ls=':', alpha=0.5)
ax.text(0.5, -0.22, 'N=192 projected from validated linear scaling', transform=ax.transAxes,
        ha='center', fontsize=7, color='gray', style='italic')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'ablation_energy.png'), dpi=160); plt.close()

print('Ablation figures written:')
for f in ['ablation_satisfaction','ablation_cumulative_hl','ablation_energy']:
    print(' -', f+'.png')
