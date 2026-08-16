"""assist (MHSPO + battery-loss term) rho_batt sweep -> target (sat~0.82, cumHL~19).

Heuristic (DOGD online learning, NO offline RL training). New physics
(isolated subclass; gold/core/MHSPO untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
Sweeps rho_batt (battery weight); higher -> more DoD-averse -> lower HL.
Each config: warmup + 1 seed x 5400 slots; records satisfaction & cumHL (x192/25).
Out: docs/assist_mhspo/
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from baselines.assist import AssistPolicy

TARGET_SAT, TARGET_HL = 0.82, 19.0
RHO_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]


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
OUT = 'docs/assist_mhspo'; os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'sweep.json')
res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def run(rho):
    tag = f'rho{rho:g}'
    if tag in res:
        print(f'  {tag}: cached  sat={res[tag]["sat"]:.3f} cumHL={res[tag]["cumHL"]:.2f}', flush=True); return
    pol = AssistPolicy(cfg, env, rho_d=1.0, rho_batt=rho, V_lyapunov=10.0)
    env.reset(phase='train')
    for _ in range(cfg.T_WARMUP):
        env.step(policy=pol)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    H = []; t0 = time.time()
    for _ in range(T):
        _, _, _, info = env.step(policy=pol)
        H.append(info['avg_health_loss'])
    sat = float(info.get('eval_satisfaction_rate', float('nan')))
    cumHL = float(np.cumsum(H)[-1] * NSAT)
    res[tag] = {'rho_batt': rho, 'sat': sat, 'cumHL': cumHL}
    json.dump(res, open(CACHE, 'w'))
    print(f'  {tag}: rho_batt={rho}  sat={sat:.3f}  cumHL={cumHL:.2f}  [{time.time()-t0:.0f}s]', flush=True)


print(f'[target] satisfaction~{TARGET_SAT}, cumHL~{TARGET_HL}  (MHSPO baseline: 0.826, 34.7)', flush=True)
for rho in RHO_GRID:
    run(rho)

print('\n=== assist (MHSPO+battery) rho_batt sweep ===', flush=True)
print(f"{'rho_batt':>10}{'sat':>8}{'cumHL':>9}{'dist':>10}", flush=True)
best = None
for rho in RHO_GRID:
    r = res[f'rho{rho:g}']
    d = ((r['sat'] - TARGET_SAT) / TARGET_SAT) ** 2 + ((r['cumHL'] - TARGET_HL) / TARGET_HL) ** 2
    print(f"{rho:10g}{r['sat']:8.3f}{r['cumHL']:9.2f}{d:10.4f}", flush=True)
    if best is None or d < best[1]:
        best = (rho, d)
print(f"\nclosest to target ({TARGET_SAT}, {TARGET_HL}): rho_batt={best[0]:g} -> "
      f"sat={res[f'rho{best[0]:g}']['sat']:.3f} cumHL={res[f'rho{best[0]:g}']['cumHL']:.2f}", flush=True)

fig, ax = plt.subplots(figsize=(7, 5))
for rho in RHO_GRID:
    r = res[f'rho{rho:g}']
    ax.scatter(r['sat'], r['cumHL'], s=60)
    ax.annotate(f"ρ={rho:g}", (r['sat'], r['cumHL']), fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.scatter([0.826], [34.7], marker='s', s=80, color='green', label='MHSPO baseline')
ax.scatter([TARGET_SAT], [TARGET_HL], marker='*', s=260, color='red', zorder=5, label='target (0.82, 19)')
ax.set_xlabel('Satisfaction'); ax.set_ylabel('Cumulative health loss')
ax.set_title('assist = MHSPO + battery-loss term (rho_batt sweep)')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'assist_mhspo_tradeoff.{ext}'), dpi=170)
plt.close()
print('[done] ->', OUT, flush=True)
