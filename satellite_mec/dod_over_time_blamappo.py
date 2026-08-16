"""BLA-MAPPO DoD distribution over time (NEW params, eval, 1 seed).

New physics (isolated subclass; gold/core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
Records per-sat per-slot DoD for the gold checkpoint, plots the across-satellite
median / P10-P90 band / P95 over time with the DoD_max=0.8 line, and prints the
requested satellite-slot statistics. Out: docs/dod_over_time/
"""
import os, time
import numpy as np
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


cfg = _C(); cfg.N_EVAL_RUNS = 1
cfg.V_DVFS = 2e17
env = SatelliteMECEnv(cfg)
NS = cfg.N_SATS; T = cfg.T_EVAL; DMAX = cfg.DOD_MAX
OUT = 'docs/dod_over_time'; os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, 'bla_dod.npy')

if os.path.exists(CACHE):
    DoD = np.load(CACHE); print('[cache] loaded DoD', DoD.shape, flush=True)
else:
    pol = MAPPOPolicy(cfg, name='LyaMAPPO'); pol.load('checkpoints/LyaMAPPO_lh4_32K')
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    sats = env.constellation.satellites
    DoD = np.empty((T, NS), np.float32); t0 = time.time()
    for t in range(T):
        env.step(policy=pol)
        DoD[t] = [s.dod for s in sats]
    np.save(CACHE, DoD)
    print(f'[run] BLA-MAPPO 1 seed x {T} slots [{time.time()-t0:.0f}s]', flush=True)

# ── across-satellite percentiles per slot ──
x = np.arange(T)
med = np.median(DoD, axis=1)
p10 = np.percentile(DoD, 10, axis=1)
p90 = np.percentile(DoD, 90, axis=1)
p95 = np.percentile(DoD, 95, axis=1)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.fill_between(x, p10, p90, color='#aec7e8', alpha=0.6, lw=0, label='P10–P90')
ax.plot(x, med, color='#1f77b4', lw=2.0, label='Median')
ax.plot(x, p95, color='#ff7f0e', lw=1.6, ls='--', label='P95')
ax.axhline(DMAX, color='red', lw=1.5, ls='--', label='DoD$_{\\max}$ = 0.8')
ax.annotate('DoD$_{\\max}$ = 0.8', xy=(T * 0.015, DMAX), xytext=(T * 0.015, DMAX - 0.055),
            color='red', fontsize=10, va='top')
ax.set_xlabel('Time slot', fontsize=12)
ax.set_ylabel('DoD', fontsize=12)
ax.set_title('BLA-MAPPO DoD Distribution Over Time', fontsize=13)
ax.set_xlim(0, T); ax.set_ylim(0, 0.85)
ax.tick_params(labelsize=10)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, ls=':', alpha=0.45)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'bla_dod_over_time.png'), dpi=300)
plt.savefig(os.path.join(OUT, 'bla_dod_over_time.pdf'))
plt.close()

# ── statistics over ALL satellite-slot samples ──
flat = DoD.ravel().astype(float)
print('\n=== BLA-MAPPO DoD statistics (all satellite-slot samples, new params, 1 seed) ===', flush=True)
print(f"  samples (sat x slot)      : {flat.size} ({NS} sats x {T} slots)", flush=True)
print(f"  fraction DoD > 0.6        : {(flat > 0.6).mean()*100:.3f}%", flush=True)
print(f"  fraction DoD > 0.7        : {(flat > 0.7).mean()*100:.3f}%", flush=True)
print(f"  max DoD                   : {flat.max():.4f}", flush=True)
print(f"  global 95th percentile    : {np.percentile(flat, 95):.4f}", flush=True)
print(f"  mean DoD                  : {flat.mean():.4f}", flush=True)
print(f"  median DoD                : {np.median(flat):.4f}", flush=True)
print('[done] figure ->', OUT, flush=True)
