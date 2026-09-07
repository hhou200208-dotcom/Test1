"""Training-budget ablation: BLA-MAPPO (our method) at 16K vs 32K training steps,
shown against MHSPO / GDCO / LSO for context.

Honest framing: 'BLA-MAPPO (16K)' is the SAME algorithm as BLA-MAPPO, just trained
for fewer steps (same reward / networks). It is NOT a separate method.

- BLA-MAPPO (32K) = gold, from committed docs/series_lh4_n25.json (key 'LyaMAPPO').
- BLA-MAPPO (16K) = this run's checkpoint, evaluated with the SAME eval seed (cached).
Outputs the 5 standard comparison figures to docs/figures_ablation_16k/.
Figures carry axis labels only (no title / no caption note — supplied in the paper).
"""
import json, os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE  = os.path.dirname(os.path.abspath(__file__))
S     = json.load(open(os.path.join(HERE, 'docs', 'series_lh4_n25.json')))
OUT   = os.path.join(HERE, 'docs', 'figures_ablation_16k'); os.makedirs(OUT, exist_ok=True)
CKPT  = os.path.join(HERE, 'results/20260725_115745_MAPPO_lh4.0_lyadrl_dod/LyaDRL_DoD/model')
CACHE = os.path.join(HERE, 'docs', 'series_bla16k.json')   # 16K 评估 series 落仓库(可提交/复现)
RATIO = 192 / 25
NSAT  = 192


def eval_16k():
    """Eval the 16K checkpoint once, record per-slot series (cached to avoid re-eval)."""
    if os.path.exists(CACHE):
        print('[16K] loaded cached series', flush=True)
        return json.load(open(CACHE))
    from core import Config, SatelliteMECEnv
    from training import MAPPOPolicy

    class _C(Config):
        LAMBDA_HIGH = 4.0
        LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    cfg = _C(); cfg.N_EVAL_RUNS = 1
    env = SatelliteMECEnv(cfg)
    pol = MAPPOPolicy(cfg, name='BLA_MAPPO_16K'); pol.load(CKPT)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    E = []; D = []; H = []; Q = []; SAT = []; DEN = []
    for _ in range(cfg.T_EVAL):
        _, _, _, info = env.step(policy=pol)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
        H.append(info['avg_health_loss']);    Q.append(info['queue_task_count'])
        SAT.append(info['slot_satisfaction_rate']); DEN.append(info['done_tasks'] + info['slot_timeout'])
    ser = {'satisfaction': float(info.get('eval_satisfaction_rate', float('nan'))),
           'energy': E, 'delay': D, 'hl': H, 'queue_tasks': Q, 'sat_slot': SAT, 'sat_denom': DEN}
    json.dump(ser, open(CACHE, 'w'))
    print(f"[16K] evaluated & cached: satisfaction={ser['satisfaction']:.3f} "
          f"HL/slot={np.mean(H)*1e4:.2f}e-4", flush=True)
    return ser


S['BLA_MAPPO_16K'] = eval_16k()

# ── plotting config ──
ORDER = [p for p in ['LyaMAPPO', 'BLA_MAPPO_16K', 'MHSPO', 'GDCO', 'LocalOnly'] if p in S]
DISP  = {'LyaMAPPO': 'BLA-MAPPO', 'BLA_MAPPO_16K': 'LyaDRL-DoD', 'LocalOnly': 'LSO'}
disp  = lambda p: DISP.get(p, p)
COLOR = {'LyaMAPPO': '#d62728', 'BLA_MAPPO_16K': '#F2C200', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}                       # 16K = 黄色实线
MARK  = {'LyaMAPPO': 's', 'BLA_MAPPO_16K': 'D', 'MHSPO': '^', 'GDCO': 'P', 'LocalOnly': '*'}
big   = lambda p: p in ('LyaMAPPO', 'BLA_MAPPO_16K')


def _finish(ax, xlabel, ylabel, fname, headroom=False):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)               # 只留横纵坐标，无标题/注释
    if headroom:                                               # 时间序列：顶部留白，右上图例不遮曲线
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi - lo) * 1.25)
    ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=160); plt.close()


def kde_fig(vals, xlabel, ylabel, fname, smooth=2.2):
    allv = np.concatenate([np.asarray(v, float) for v in vals.values()]); allv = allv[np.isfinite(allv)]
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in ORDER:
        v = np.asarray(vals[p], float); v = v[np.isfinite(v)]
        if len(v) < 10 or v.std() < 1e-9:
            continue
        kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * smooth); ys = kde(xs)
        ax.plot(xs, ys, color=COLOR[p], lw=2.4 if big(p) else 1.6, zorder=3 if big(p) else 2)
        mk = np.linspace(0, len(xs) - 1, 18).astype(int)
        ax.plot(xs[mk], ys[mk], color=COLOR[p], marker=MARK[p], ls='none',
                markersize=6 if big(p) else 5, label=disp(p), zorder=3 if big(p) else 2)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    _finish(ax, xlabel, ylabel, fname)


kde_fig({p: np.asarray(S[p]['delay']) * RATIO for p in ORDER},
        'System total delay', 'Probability density', 'delay_pdf.png')
kde_fig({p: np.asarray(S[p]['energy']) * RATIO / 1e3 for p in ORDER},
        'System total energy', 'Probability density', 'energy_pdf.png')
sat = {}
for p in ORDER:
    s = np.asarray(S[p]['sat_slot'], float); d = np.asarray(S[p]['sat_denom'], float); sat[p] = s[d > 0]
kde_fig(sat, 'User satisfaction', 'Probability density', 'satisfaction_pdf.png')

# ── cumulative HL (time series) ──
fig, ax = plt.subplots(figsize=(7, 5))
for p in ORDER:
    y = np.cumsum(np.asarray(S[p]['hl'], float)) * NSAT
    ax.plot(np.arange(len(y)), y, color=COLOR[p], lw=2.4 if big(p) else 1.5,
            label=disp(p), zorder=3 if big(p) else 2)
_finish(ax, 'Time slot', 'System total health loss', 'cumulative_hl.png', headroom=True)

# ── queue length (task count, time series, 150-slot moving average) ──
def _smooth(a, w=150):
    a = np.asarray(a, float); return np.convolve(a, np.ones(w) / w, mode='valid')

fig, ax = plt.subplots(figsize=(7, 5))
for p in ORDER:
    q = _smooth(np.asarray(S[p]['queue_tasks']) * NSAT, 150)
    ax.plot(np.arange(len(q)), q, color=COLOR[p], lw=2.4 if big(p) else 1.4,
            label=disp(p), zorder=3 if big(p) else 2)
_finish(ax, 'Time slot', 'System queue length', 'queue_backlog.png', headroom=True)

print('figures written to', OUT, flush=True)
for f in ['satisfaction_pdf', 'energy_pdf', 'delay_pdf', 'cumulative_hl', 'queue_backlog']:
    print(' -', f + '.png', flush=True)
