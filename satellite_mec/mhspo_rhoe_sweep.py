"""MHSPO energy-weight (rho_e) sweep -> find a point with energy < w/o Task Prior
(0.2665 kJ/slot) AND cumHL > w/o Task Prior (11.22).

Heuristic (DOGD online learning, no offline training). New physics
(isolated subclass; core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD[0,0.8].
Higher rho_e -> MHSPO penalizes compute energy more -> lower system energy.
Each config: warmup + 1 seed x 5400 slots; records energy(kJ/slot,x192/25),
cumHL(x192), satisfaction. Out: docs/mhspo_rhoe/
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from baselines.mhspo import MHSPOPolicy

REF_E, REF_HL = 0.2665, 11.22          # w/o Task Prior reference
RHO_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0


cfg = _C(); cfg.N_EVAL_RUNS = 1; cfg.V_DVFS = 2e17
env = SatelliteMECEnv(cfg)
T = cfg.T_EVAL; RATIO = 192.0 / cfg.N_SATS; NSAT = 192
OUT = 'docs/mhspo_rhoe'; os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'sweep.json')
res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def run(rho):
    tag = f'rhoe{rho:g}'
    if tag in res:
        print(f'  {tag}: cached E={res[tag]["E"]:.4f} HL={res[tag]["cumHL"]:.2f}', flush=True); return
    pol = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=rho, V_lyapunov=10.0)
    env.reset(phase='train')
    for _ in range(cfg.T_WARMUP):
        env.step(policy=pol)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    E = []; H = []; t0 = time.time()
    for _ in range(T):
        _, _, _, info = env.step(policy=pol)
        E.append(info['slot_system_energy']); H.append(info['avg_health_loss'])
    sat = float(info.get('eval_satisfaction_rate', float('nan')))
    Ek = float(np.mean(E) * RATIO / 1e3); cumHL = float(np.cumsum(H)[-1] * NSAT)
    res[tag] = {'rho_e': rho, 'E': Ek, 'cumHL': cumHL, 'sat': sat}
    json.dump(res, open(CACHE, 'w'))
    ok = 'HIT' if (Ek < REF_E and cumHL > REF_HL) else ''
    print(f'  {tag}: rho_e={rho:g}  E={Ek:.4f}  cumHL={cumHL:.2f}  sat={sat:.3f}  {ok}', flush=True)


print(f'[target] energy < {REF_E} AND cumHL > {REF_HL}  (w/o Task Prior ref)', flush=True)
for rho in RHO_GRID:
    run(rho)

print('\n=== MHSPO rho_e sweep (new params, 1 seed) ===', flush=True)
print(f"{'rho_e':>7}{'energy':>9}{'cumHL':>8}{'sat':>7}  {'E<ref':>6} {'HL>ref':>7} {'BOTH':>5}", flush=True)
hits = []
for rho in RHO_GRID:
    r = res[f'rhoe{rho:g}']
    e_ok = r['E'] < REF_E; h_ok = r['cumHL'] > REF_HL; both = e_ok and h_ok
    if both:
        hits.append(rho)
    print(f"{rho:7g}{r['E']:9.4f}{r['cumHL']:8.2f}{r['sat']:7.3f}  "
          f"{'Y' if e_ok else '-':>6} {'Y' if h_ok else '-':>7} {'YES' if both else '':>5}", flush=True)
print(f"\nconfigs satisfying BOTH (E<{REF_E}, HL>{REF_HL}): {hits if hits else 'NONE'}", flush=True)

# tradeoff plot
fig, ax = plt.subplots(figsize=(7, 5))
for rho in RHO_GRID:
    r = res[f'rhoe{rho:g}']
    ax.scatter(r['E'], r['cumHL'], s=60, color='#2ca02c')
    ax.annotate(f"ρe={rho:g}", (r['E'], r['cumHL']), fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.scatter([REF_E], [REF_HL], marker='*', s=260, color='#8c564b', zorder=5, label='w/o Task Prior (0.266,11.2)')
ax.axvline(REF_E, color='gray', ls='--', lw=1); ax.axhline(REF_HL, color='gray', ls='--', lw=1)
ax.set_xlabel('System energy (kJ/slot)'); ax.set_ylabel('Cumulative health loss')
ax.set_title('MHSPO rho_e sweep vs w/o Task Prior (target: upper-left quadrant)')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'mhspo_rhoe_tradeoff.{ext}'), dpi=170)
plt.close()
print('[done] ->', OUT, flush=True)
