"""3-method ablation figures (new params): BLA-MAPPO / w/o DoD / w/o Task Prior.

Option (C): honest converged checkpoints.
  BLA-MAPPO     (red)   = gold LyaMAPPO_lh4_32K   (series reused from cache)
  w/o DoD       (pink)  = MAPPO_NoBat_lh4_8K       (DoD ablation; 8K budget caveat)
  w/o Task Prior(brown) = wo_task_prior_16k/step_12000 (BETA_TASK=0; converged-region,
                          NOT the cherry-picked step_2000)
New physics: kappa=1.5e-27, E_cap=54kJ, V_f=2e17, DoD[0,0.8]. 1 seed.
4 figs: satisfaction PDF, delay PDF, energy PDF, cumulative HL. Out: docs/ablation3/
"""
import os, json, time
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0

cfg = _C(); cfg.N_EVAL_RUNS = 1; cfg.V_DVFS = 2e17
env = SatelliteMECEnv(cfg)
T = cfg.T_EVAL; RATIO = 192.0 / cfg.N_SATS; NSAT = 192
OUT = 'docs/ablation3'; os.makedirs(OUT, exist_ok=True)


def eval_ckpt(path):
    p = MAPPOPolicy(cfg, name='e'); p.load(path)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); p.set_eval_mode()
    E = []; D = []; H = []; SAT = []; DEN = []
    for _ in range(T):
        _, _, _, info = env.step(policy=p)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
        H.append(info['avg_health_loss']); SAT.append(info['slot_satisfaction_rate'])
        DEN.append(info['done_tasks'] + info['slot_timeout'])
    return {'E': E, 'D': D, 'hl': H, 'sat_slot': SAT, 'sat_denom': DEN,
            'sat': float(info.get('eval_satisfaction_rate', float('nan')))}


S = {}
# BLA-MAPPO from cached new-params series
cache = json.load(open('docs/newparams_series/series.json'))
S['BLA-MAPPO'] = cache['LyaMAPPO']
print(f"BLA-MAPPO (cached): cumHL={np.cumsum(S['BLA-MAPPO']['hl'])[-1]*NSAT:.2f}", flush=True)
for name, ck in [('w/o DoD', 'checkpoints/MAPPO_NoBat_lh4_8K'),
                 ('w/o Task Prior', 'checkpoints/wo_task_prior_16k/step_12000')]:
    t0 = time.time(); S[name] = eval_ckpt(ck)
    print(f"{name}: sat={S[name]['sat']:.3f} cumHL={np.cumsum(S[name]['hl'])[-1]*NSAT:.2f} [{time.time()-t0:.0f}s]", flush=True)

ORDER = ['BLA-MAPPO', 'w/o DoD', 'w/o Task Prior']
COLOR = {'BLA-MAPPO': '#d62728', 'w/o DoD': '#e377c2', 'w/o Task Prior': '#8c564b'}
MARK = {'BLA-MAPPO': 's', 'w/o DoD': 'o', 'w/o Task Prior': 'P'}
big = lambda k: k == 'BLA-MAPPO'


def kde_fig(vals, xlabel, fname, smooth=2.2):
    allv = np.concatenate([np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for v in vals.values()])
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
                markersize=6 if big(k) else 5, label=k, zorder=3 if big(k) else 2)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel); ax.set_ylabel('Probability density')
    ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUT, f'{fname}.{ext}'), dpi=175)
    plt.close()


sat = {}
for k in ORDER:
    s = np.asarray(S[k]['sat_slot'], float); d = np.asarray(S[k]['sat_denom'], float); sat[k] = s[d > 0]
kde_fig(sat, 'User satisfaction', 'abl3_satisfaction')
kde_fig({k: np.asarray(S[k]['D']) * RATIO for k in ORDER}, 'System total delay (s)', 'abl3_delay')
kde_fig({k: np.asarray(S[k]['E']) * RATIO / 1e3 for k in ORDER}, 'System total energy', 'abl3_energy')

fig, ax = plt.subplots(figsize=(7, 5))
for k in ORDER:
    y = np.cumsum(np.asarray(S[k]['hl'], float)) * NSAT
    ax.plot(np.arange(T), y, color=COLOR[k], lw=2.4 if big(k) else 1.6, label=k, zorder=3 if big(k) else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'abl3_cumulative_hl.{ext}'), dpi=175)
plt.close()
print('[done] 4 figs ->', OUT, flush=True)
