"""System cumulative health-loss comparison UNDER NEW PARAMS (eval, 1 seed).

New physics (isolated config subclass; gold/core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
Metric: TANGENT (validated correct) = sum_t sum_n L'(DoD)*(dDoD_comp+dDoD_trans),
i.e. cumsum(avg_health_loss) * 192/25 — same as paper fig:hl.
CAVEAT: RL methods (BLA-MAPPO, LyDRL-DoD) trained on OLD physics -> off-distribution;
figure shows shape/trend, absolute RL HL is not a fair number without retrain.
Out: docs/hl_newparams/
"""
import os, time, csv
import numpy as np
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
T = cfg.T_EVAL; NSAT = 192
OUT = 'docs/hl_newparams'; os.makedirs(OUT, exist_ok=True)
print(f'[cfg] kappa={cfg.KAPPA:.1e} E_cap={cfg.E_CAP:.0f} V_DVFS={cfg.V_DVFS:.1e} DoD_min={cfg.DOD_MIN}', flush=True)


def _load(p, path):
    p.load(path); return p


METHODS = [
    ('LyaMAPPO',   lambda: _load(MAPPOPolicy(cfg, name='LyaMAPPO'), 'checkpoints/LyaMAPPO_lh4_32K'), False),
    ('MADDPG_DoD', lambda: _load(MADDPGDoDPolicy(cfg, env), 'checkpoints/MADDPG_DoD_lh4_32K'), False),
    ('MHSPO',      lambda: MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0), True),
    ('GDCO',       lambda: GDCOPolicy(cfg, env), False),
    ('LocalOnly',  lambda: LocalOnlyPolicy(cfg, env), False),
]

HL = {}; SAT = {}
for key, make, warm in METHODS:
    pol = make()
    if warm:
        env.reset(phase='train')
        for _ in range(cfg.T_WARMUP):
            env.step(policy=pol)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    h = []; t0 = time.time()
    for _ in range(T):
        _, _, _, info = env.step(policy=pol)
        h.append(info['avg_health_loss'])
    HL[key] = np.asarray(h, float)
    SAT[key] = float(info.get('eval_satisfaction_rate', float('nan')))
    print(f'  {key}: cumHL={np.cumsum(h)[-1]*NSAT:.2f}  sat={SAT[key]:.3f}  [{time.time()-t0:.0f}s]', flush=True)

# ── plot ──
ORDER = ['LyaMAPPO', 'MADDPG_DoD', 'MHSPO', 'GDCO', 'LocalOnly']
DISP = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD', 'LocalOnly': 'LSO'}
COLOR = {'LyaMAPPO': '#d62728', 'MADDPG_DoD': '#F2C200', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}
fig, ax = plt.subplots(figsize=(7.5, 5))
rows = []
for k in ORDER:
    y = np.cumsum(HL[k]) * NSAT; big = (k == 'LyaMAPPO')
    ax.plot(np.arange(T), y, color=COLOR[k], lw=2.4 if big else 1.5,
            label=DISP.get(k, k), zorder=3 if big else 2)
    rows.append({'method': DISP.get(k, k), 'cum_health_loss': float(y[-1]), 'satisfaction': SAT[k]})
ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
ax.legend(fontsize=9, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.set_title('New params (kappa=1.5e-27, E_cap=54kJ, V_f=2e17, DoD[0,0.8]) — 1 seed', fontsize=9)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'cumulative_hl_newparams.{ext}'), dpi=175)
plt.close()

order = sorted(rows, key=lambda r: r['cum_health_loss'])
with open(os.path.join(OUT, 'cumulative_hl_newparams.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['method', 'cum_health_loss', 'satisfaction']); w.writeheader()
    for r in rows:
        w.writerow(r)
print('\n=== cumHL ranking (low=better) [NEW PARAMS, 1 seed] ===', flush=True)
for r in order:
    print(f"  {r['method']:11} cumHL={r['cum_health_loss']:10.2f}  sat={r['satisfaction']:.3f}", flush=True)
print('[done] ->', OUT, flush=True)
