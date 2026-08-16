"""5-method per-slot SYSTEM ENERGY distribution under NEW params (eval, 1 seed).

New physics (isolated subclass; gold/core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
Records slot_system_energy + delay + satisfaction + HL for all 5 methods and
caches them (docs/newparams_series/series.json) so future distribution figures
need no re-run. Produces the system energy PDF with the SAME colors as the
cumulative-HL comparison figure.
CAVEAT: RL methods off-distribution; 1 seed -> shape only. Out: docs/newparams_series/
"""
import os, json, time
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.maddpg_dod import MADDPGDoDPolicy
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
T = cfg.T_EVAL; RATIO = 192.0 / cfg.N_SATS
OUT = 'docs/newparams_series'; os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'series.json')
data = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

# colors identical to the cumulative-HL comparison figure (hl_cum_newparams.py)
ORDER = ['LyaMAPPO', 'MADDPG_DoD', 'MHSPO', 'GDCO', 'LocalOnly']
DISP = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD', 'LocalOnly': 'LSO'}
disp = lambda k: DISP.get(k, k)
COLOR = {'LyaMAPPO': '#d62728', 'MADDPG_DoD': '#F2C200', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}
MARK = {'LyaMAPPO': 's', 'MADDPG_DoD': 'D', 'MHSPO': '^', 'GDCO': 'P', 'LocalOnly': '*'}
big = lambda k: k == 'LyaMAPPO'


def evalm(pol, key, warm=False):
    if key in data:
        print(f'  {key}: cached', flush=True); return
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
    data[key] = {'E': E, 'D': D, 'hl': H, 'sat_slot': SAT, 'sat_denom': DEN}
    json.dump(data, open(CACHE, 'w'))
    print(f"  {key}: energy_mean={np.mean(E)*RATIO/1e3:.3f}kJ/slot [{time.time()-t0:.0f}s]", flush=True)


_ld = lambda p, path: (p.load(path), p)[1]
evalm(_ld(MAPPOPolicy(cfg, name='LyaMAPPO'), 'checkpoints/LyaMAPPO_lh4_32K'), 'LyaMAPPO')
evalm(_ld(MADDPGDoDPolicy(cfg, env), 'checkpoints/MADDPG_DoD_lh4_32K'), 'MADDPG_DoD')
evalm(MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0), 'MHSPO', warm=True)
evalm(GDCOPolicy(cfg, env), 'GDCO')
evalm(LocalOnlyPolicy(cfg, env), 'LocalOnly')

# ── system energy PDF ──
vals = {k: np.asarray(data[k]['E'], float) * RATIO / 1e3 for k in ORDER}   # kJ/slot, x192/25
allv = np.concatenate([v[np.isfinite(v)] for v in vals.values()])
lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
xs = np.linspace(lo, hi, 400)
fig, ax = plt.subplots(figsize=(7, 5))
for k in ORDER:
    v = vals[k]; v = v[np.isfinite(v)]
    if len(v) < 10 or v.std() < 1e-9:
        continue
    kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * 2.2); ys = kde(xs)
    ax.plot(xs, ys, color=COLOR[k], lw=2.4 if big(k) else 1.6, zorder=3 if big(k) else 2)
    mk = np.linspace(0, len(xs) - 1, 16).astype(int)
    ax.plot(xs[mk], ys[mk], color=COLOR[k], marker=MARK[k], ls='none',
            markersize=6 if big(k) else 5, label=disp(k), zorder=3 if big(k) else 2)
ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)
ax.set_xlabel('System total energy'); ax.set_ylabel('Probability density')
ax.legend(fontsize=9, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'energy_pdf_newparams.{ext}'), dpi=175)
plt.close()
print('[done] energy PDF ->', OUT, flush=True)
for k in ORDER:
    print(f"  {disp(k):11} energy mean={np.mean(data[k]['E'])*RATIO/1e3:.3f} kJ/slot", flush=True)
