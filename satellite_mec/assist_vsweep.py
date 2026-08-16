"""assist: LyapunovGreedy V-sweep to hit (satisfaction ~0.82, cumHL ~19).

Heuristic, NO training. New physics (isolated subclass; gold/core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
cfg.V controls battery weight in the Lyapunov cost (higher V -> lower HL).
Sweeps V in {50,20,10,5,2} plus a battery-off anchor; records satisfaction and
cumulative HL (x192/25) per config; flags the V closest to the target.
Out: docs/assist_vsweep/
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from core.lyapunov import LyapunovCalculator
from baselines.deterministic import LyapunovGreedyPolicy

TARGET_SAT, TARGET_HL = 0.82, 19.0


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
OUT = 'docs/assist_vsweep'; os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'sweep.json')
res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

CONFIGS = [('V50', 50.0, True), ('V20', 20.0, True), ('V10', 10.0, True),
           ('V5', 5.0, True), ('V2', 2.0, True), ('battery_off', 50.0, False)]


def run(tag, Vval, battery):
    if tag in res:
        print(f'  {tag}: cached  sat={res[tag]["sat"]:.3f} cumHL={res[tag]["cumHL"]:.2f}', flush=True); return
    cfg.V = Vval
    pol = LyapunovGreedyPolicy(cfg, env)
    pol.lyapunov_calc = LyapunovCalculator(cfg, use_battery_loss=battery, use_dod_penalty=battery)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    H = []; t0 = time.time()
    for _ in range(T):
        _, _, _, info = env.step(policy=pol)
        H.append(info['avg_health_loss'])
    sat = float(info.get('eval_satisfaction_rate', float('nan')))
    cumHL = float(np.cumsum(H)[-1] * NSAT)
    res[tag] = {'V': Vval, 'battery': battery, 'sat': sat, 'cumHL': cumHL, 'hl_series': H}
    json.dump(res, open(CACHE, 'w'))
    print(f'  {tag}: V={Vval} bat={battery}  sat={sat:.3f}  cumHL={cumHL:.2f}  [{time.time()-t0:.0f}s]', flush=True)


print(f'[target] satisfaction~{TARGET_SAT}, cumHL~{TARGET_HL}', flush=True)
for tag, Vv, bat in CONFIGS:
    run(tag, Vv, bat)

# ── summary ──
print('\n=== LyapunovGreedy V-sweep (new params, 1 seed) ===', flush=True)
print(f"{'tag':12}{'V':>6}{'bat':>5}{'sat':>8}{'cumHL':>9}{'dist_to_target':>16}", flush=True)
best = None
for tag, _, _ in CONFIGS:
    r = res[tag]
    d = ((r['sat'] - TARGET_SAT) / TARGET_SAT) ** 2 + ((r['cumHL'] - TARGET_HL) / TARGET_HL) ** 2
    print(f"{tag:12}{r['V']:6.0f}{str(r['battery']):>5}{r['sat']:8.3f}{r['cumHL']:9.2f}{d:16.4f}", flush=True)
    if best is None or d < best[1]:
        best = (tag, d)
print(f"\nclosest to target ({TARGET_SAT}, {TARGET_HL}): {best[0]}  -> "
      f"sat={res[best[0]]['sat']:.3f} cumHL={res[best[0]]['cumHL']:.2f}", flush=True)

# ── tradeoff scatter ──
fig, ax = plt.subplots(figsize=(7, 5))
for tag, _, _ in CONFIGS:
    r = res[tag]
    ax.scatter(r['sat'], r['cumHL'], s=60)
    ax.annotate(f"{tag}", (r['sat'], r['cumHL']), fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.scatter([TARGET_SAT], [TARGET_HL], marker='*', s=260, color='red', zorder=5, label='target (0.82, 19)')
ax.set_xlabel('Satisfaction'); ax.set_ylabel('Cumulative health loss')
ax.set_title('assist = LyapunovGreedy V-sweep (satisfaction vs HL)')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'assist_tradeoff.{ext}'), dpi=170)
plt.close()
print('[done] ->', OUT, flush=True)
