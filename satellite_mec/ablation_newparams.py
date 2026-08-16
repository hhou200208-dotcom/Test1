"""4 figures for BLA-MAPPO / GDCO / MAPPO-NoDoD under NEW params (eval, 1 seed).

New physics (isolated subclass; gold/core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
Figures (paper conventions, x192/25 projection for extensive quantities):
  1 satisfaction PDF   2 system delay PDF   3 system energy PDF   4 cumulative HL
Colors: BLA-MAPPO red, GDCO brown, MAPPO-NoDoD pink.
CAVEAT: RL methods off-distribution (old-physics policies); MAPPO-NoDoD only 8K-trained;
1 seed -> shape only, not fair final numbers. Out: docs/ablation_newparams/
"""
import os, time
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from baselines import GDCOPolicy
from training import MAPPOPolicy


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0


cfg = _C(); cfg.N_EVAL_RUNS = 1
cfg.V_DVFS = 2e17
env = SatelliteMECEnv(cfg)
T = cfg.T_EVAL; RATIO = 192.0 / cfg.N_SATS; NSAT = 192
OUT = 'docs/ablation_newparams'; os.makedirs(OUT, exist_ok=True)
ORDER = ['LyaMAPPO', 'GDCO', 'MAPPO_NoBat']
DISP = {'LyaMAPPO': 'BLA-MAPPO', 'MAPPO_NoBat': 'MAPPO-NoDoD'}
disp = lambda k: DISP.get(k, k)
COLOR = {'LyaMAPPO': '#d62728', 'GDCO': '#8c564b', 'MAPPO_NoBat': '#e377c2'}
MARK = {'LyaMAPPO': 's', 'GDCO': 'P', 'MAPPO_NoBat': 'o'}
big = lambda k: k == 'LyaMAPPO'


def evalm(pol, key, warm=False):
    if warm:
        env.reset(phase='train')
        for _ in range(cfg.T_WARMUP):
            env.step(policy=pol)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    E = []; D = []; H = []; SAT = []; DEN = []; t0 = time.time()
    for _ in range(T):
        _, _, _, info = env.step(policy=pol)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
        H.append(info['avg_health_loss']); SAT.append(info['slot_satisfaction_rate'])
        DEN.append(info['done_tasks'] + info['slot_timeout'])
    print(f"  {key}: cumHL={np.cumsum(H)[-1]*NSAT:.2f} sat={info.get('eval_satisfaction_rate',float('nan')):.3f} [{time.time()-t0:.0f}s]", flush=True)
    return {'E': E, 'D': D, 'hl': H, 'sat_slot': SAT, 'sat_denom': DEN}


S = {}
S['LyaMAPPO'] = evalm((lambda p: (p.load('checkpoints/LyaMAPPO_lh4_32K'), p)[1])(MAPPOPolicy(cfg, name='LyaMAPPO')), 'LyaMAPPO')
S['GDCO'] = evalm(GDCOPolicy(cfg, env), 'GDCO')
S['MAPPO_NoBat'] = evalm((lambda p: (p.load('checkpoints/MAPPO_NoBat_lh4_8K'), p)[1])(MAPPOPolicy(cfg, name='MAPPO_NoBat')), 'MAPPO_NoBat')


def kde_fig(vals, xlabel, fname, smooth=2.2):
    allv = np.concatenate([np.asarray(v, float) for v in vals.values()]); allv = allv[np.isfinite(allv)]
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in ORDER:
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
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUT, f'{fname}.{ext}'), dpi=175)
    plt.close()


# 1 satisfaction PDF (rate, no projection)
sat = {}
for k in ORDER:
    s = np.asarray(S[k]['sat_slot'], float); d = np.asarray(S[k]['sat_denom'], float); sat[k] = s[d > 0]
kde_fig(sat, 'User satisfaction', 'ablation_satisfaction_newparams')
# 2 delay PDF (extensive, xRATIO)
kde_fig({k: np.asarray(S[k]['D']) * RATIO for k in ORDER}, 'System total delay (s)', 'ablation_delay_newparams')
# 3 energy PDF (extensive, xRATIO, kJ->/1e3)
kde_fig({k: np.asarray(S[k]['E']) * RATIO / 1e3 for k in ORDER}, 'System total energy', 'ablation_energy_newparams')

# 4 cumulative HL (xNSAT)
fig, ax = plt.subplots(figsize=(7, 5))
for k in ORDER:
    y = np.cumsum(np.asarray(S[k]['hl'], float)) * NSAT
    ax.plot(np.arange(T), y, color=COLOR[k], lw=2.4 if big(k) else 1.6, label=disp(k), zorder=3 if big(k) else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'ablation_cumulative_hl_newparams.{ext}'), dpi=175)
plt.close()
print('[done] 4 figures ->', OUT, flush=True)
