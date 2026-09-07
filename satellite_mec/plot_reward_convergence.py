"""BLA-MAPPO reward convergence — presentation-ready (fast, no retrain).

Uses existing conv3 per-slot reward (~32000 pts). Per-episode-magnitude reward
(running mean of per-slot x K), single blue line, widescreen for slides.
Two smoothing levels; pick the nicer one. Output docs/figures_convergence/.
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RC = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob(os.path.join(HERE, 'results/*conv3*/BLA_MAPPO/reward_curve.json')))[-1]
OUT = os.path.join(HERE, 'docs', 'figures_convergence'); os.makedirs(OUT, exist_ok=True)
K = 64
BLUE = '#1f77b4'

r = np.asarray(json.load(open(RC)), float)     # ~32000 per-slot rewards
ep_x = np.arange(len(r)) / K
NE = len(r) / K


def mov(x, a, w):
    y = np.convolve(a, np.ones(w) / w, mode='valid')
    return x[w // 2: w // 2 + len(y)], y


def fig(window, fname, lw):
    sx, sy = mov(ep_x, r, window)
    sy = sy * K                                 # per-episode reward magnitude
    f, ax = plt.subplots(figsize=(8, 4.5))      # widescreen for slides
    ax.plot(sx, sy, color=BLUE, lw=lw)
    ax.set_xlabel('Episode', fontsize=12); ax.set_ylabel('Reward', fontsize=12)
    ax.set_xlim(-20, NE + 15)                   # left blank margin
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.tick_params(labelsize=11)
    ax.grid(True, ls=':', alpha=0.35)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=200); plt.close()


fig(700,  'reward_convergence.png',          2.0)   # smooth (~11 ep window)
fig(1600, 'reward_convergence_smoother.png', 2.2)   # very smooth (~25 ep window)
print('written both to', OUT)
