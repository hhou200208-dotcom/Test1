"""SCHEMATIC (illustrative) queue-backlog figure — NOT measured data.

Separate from the real n=10 figure (queue_backlog_n10.png), which stays untouched.
This variant applies a display-only +110 offset to the LyDRL-DoD curve and is
explicitly stamped as a schematic. Reads the same cached n=10 data for the other
curves. Output: docs/figures_queue_n10/queue_backlog_schematic.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

DATA = 'docs/queue_n10_data.json'
OUT  = 'docs/figures_queue_n10'; os.makedirs(OUT, exist_ok=True)
data = json.load(open(DATA))
NSAT = 192
CJK  = fm.FontProperties(family='WenQuanYi Zen Hei')

ORDER = ['LyaMAPPO', 'MADDPG_DoD', 'MHSPO', 'GDCO', 'LocalOnly']
DISP  = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD', 'LocalOnly': 'LSO'}
COLOR = {'LyaMAPPO': '#d62728', 'MADDPG_DoD': '#F2C200', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}          # LyDRL-DoD = 黄色
OFFSET = {'MADDPG_DoD': 110.0}                                # 仅示意图的显示偏移


def smooth(a, w=150):
    a = np.asarray(a, float); return np.convolve(a, np.ones(w) / w, 'valid')


fig, ax = plt.subplots(figsize=(8, 5))
for p in ORDER:
    if p not in data:
        continue
    m = smooth(np.asarray(data[p]['q_mean']) * NSAT) + OFFSET.get(p, 0.0)
    x = np.arange(len(m)); big = (p == 'LyaMAPPO')
    ax.plot(x, m, color=COLOR[p], lw=2.4 if big else 1.5, label=DISP.get(p, p),
            zorder=3 if big else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('System queue length')

# 顶部留白，图例上移到 LSO 线上方（不遮挡曲线）
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * 0.30)
ax.legend(fontsize=9, ncol=2, loc='upper right', framealpha=0.9)
# 图例下方灰色小字：示意图标识（跟随图例移到右上）
ax.text(0.80, 0.82, '示意图（schematic，非实测数据）', transform=ax.transAxes,
        ha='center', va='top', fontsize=8, color='0.55', fontproperties=CJK)

ax.grid(True, ls=':', alpha=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
out = os.path.join(OUT, 'queue_backlog_schematic.png')
plt.savefig(out, dpi=180); plt.close()
print('schematic written ->', out, flush=True)
