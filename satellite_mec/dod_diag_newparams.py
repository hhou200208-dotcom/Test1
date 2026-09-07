"""Quick DoD-saturation probe under NEW physics params (eval gold, 1 seed).

Isolated config subclass — does NOT touch core/config.py or the gold checkpoint.
  DoD projection [0, 0.8]   (DOD_MIN 0.1 -> 0.0)
  kappa 1.5e-27             (was 1e-26)
  E_cap 54 kJ = 15 Wh       (was 36 kJ)
  V_f (V_DVFS) 2e17         (hard override; auto-calib with new kappa would be 5e17)
Compares against the OLD run (pooled DoD==0.8 = 14.08%, high-load 33.76%).
Read-only external logging; delta reconstruction self-validated. Out: docs/dod_diag_newparams/
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from training import MAPPOPolicy

LN10 = np.log(10.0)


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    KAPPA = 1.5e-27
    E_CAP = 54_000.0            # 15 Wh
    DOD_MIN = 0.0


cfg = _C(); cfg.N_EVAL_RUNS = 1
cfg.V_DVFS = 2e17              # hard override per user
env = SatelliteMECEnv(cfg)
NS = cfg.N_SATS; T = cfg.T_EVAL
TAU = cfg.TAU; ECAP = cfg.E_CAP; DMIN = cfg.DOD_MIN; DMAX = cfg.DOD_MAX; A = cfg.A_COEF
FMAX = cfg.CPU_FREQ; TOL = 1e-6
OUT = 'docs/dod_diag_newparams'; os.makedirs(OUT, exist_ok=True)
HIGH = np.array(sorted(env.constellation.high_load_sats))
LOW = np.array(sorted(env.constellation.low_load_sats))
print(f'[cfg] kappa={cfg.KAPPA:.2e} E_cap={ECAP:.0f}J DoD_min={DMIN} V_DVFS={cfg.V_DVFS:.2e}', flush=True)
print(f'[cfg] peak comp power kappa*c_n^3 = {cfg.KAPPA*cfg.CPU_FREQ**3:.2f} W  (old was 80 W)', flush=True)

lp = lambda d: (10.0 ** (A * (d - 1.0))) * (1.0 + A * LN10 * d)

pol = MAPPOPolicy(cfg, name='LyaMAPPO'); pol.load('checkpoints/LyaMAPPO_lh4_32K')
env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
sats = env.constellation.satellites
prev = np.array([s.dod for s in sats], float)
DoD = np.empty((T, NS), np.float32); F = np.empty((T, NS), np.float32); CP = np.empty((T, NS), np.float32)
maxerr = 0.0; t0 = time.time()
for t in range(T):
    env.step(policy=pol)
    dod = np.array([s.dod for s in sats], float)
    fcmp = np.array([s.last_cpu_freq for s in sats], float)
    ce = np.array([getattr(s, 'slot_comp_energy', 0.0) for s in sats], float)
    te = np.array([getattr(s, 'slot_trans_energy', 0.0) for s in sats], float)
    he = np.array([getattr(s, 'slot_house_energy', 0.0) for s in sats], float)
    so = np.array([s.solar_power for s in sats], float)
    dsolar = np.minimum(so * TAU / ECAP, np.maximum(prev - DMIN, 0.0))
    draw = (ce + te + he) / ECAP - dsolar
    maxerr = max(maxerr, float(np.abs(np.clip(prev + draw, DMIN, DMAX) - dod).max()))
    DoD[t] = dod; F[t] = fcmp; CP[t] = ce / TAU
    prev = dod
print(f'[run] 1 seed x {T} slots [{time.time()-t0:.0f}s]  recon max|err|={maxerr:.2e}', flush=True)

at_hi = np.abs(DoD - DMAX) <= TOL
at_lo = np.abs(DoD - DMIN) <= TOL
lpv = lp(DoD)
def pct(a, p): return float(np.percentile(a, p))

print('\n=== NEW PARAMS — DoD saturation (1 seed) ===', flush=True)
print(f"  DoD==0.8 frac = {at_hi.mean()*100:.2f}%   (OLD 36kJ/1e-26 was 14.08%)", flush=True)
print(f"  DoD==0.0 frac = {at_lo.mean()*100:.2f}%", flush=True)
print(f"  DoD mean={DoD.mean():.3f} std={DoD.std():.3f} "
      f"P50={pct(DoD,50):.3f} P90={pct(DoD,90):.3f} P95={pct(DoD,95):.3f} max={DoD.max():.3f}", flush=True)
print(f"  high-load 0.8 frac = {(np.abs(DoD[:,HIGH]-DMAX)<=TOL).mean()*100:.2f}%  (OLD 33.76%)", flush=True)
print(f"  low-load  0.8 frac = {(np.abs(DoD[:,LOW]-DMAX)<=TOL).mean()*100:.2f}%  (OLD 9.15%)", flush=True)
# longest run at 0.8
def longest(mask):
    b = c = 0
    for v in mask:
        c = c + 1 if v else 0; b = max(b, c)
    return b
runs = [longest(at_hi[:, n]) for n in range(NS)]
print(f"  longest run at 0.8: global max={max(runs)} slots (OLD 2622)", flush=True)
print(f"  f_cmp mean={F.mean()/1e9:.3f} GHz  >=0.95c_n frac={ (F>=0.95*FMAX).mean()*100:.2f}%", flush=True)
print(f"  comp power mean={CP.mean():.2f}W P95={pct(CP,95):.2f}W max={CP.max():.2f}W (peak cap {cfg.KAPPA*FMAX**3:.0f}W)", flush=True)
print(f"  L'(visited): mean={lpv.mean():.3f} std={lpv.std():.3f} min={lpv.min():.3f} max={lpv.max():.3f} "
      f"CV={lpv.std()/max(lpv.mean(),1e-9):.3f}  (OLD mean0.909 CV0.58)", flush=True)

# figures
x = np.arange(T)
med = np.median(DoD, 1); p10 = np.percentile(DoD, 10, 1); p90 = np.percentile(DoD, 90, 1); p95 = np.percentile(DoD, 95, 1)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(x, p10, p90, color='#1f77b4', alpha=0.18, lw=0, label='P10–P90')
ax.plot(x, med, color='#1f77b4', lw=1.8, label='Median')
ax.plot(x, p95, color='#ff7f0e', lw=1.2, label='P95')
ax.axhline(DMAX, color='red', ls='--', lw=1.2, label='0.8'); ax.axhline(DMIN, color='gray', ls=':', lw=1.0, label='0.0')
ax.set_xlabel('Time slot'); ax.set_ylabel('DoD'); ax.set_ylim(-0.02, 0.85)
ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_title(f'NEW params (kappa=1.5e-27, E_cap=54kJ, V_f=2e17, DoD[0,0.8]) — DoD==0.8: {at_hi.mean()*100:.1f}%', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'dod_timeseries_newparams.png'), dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist(DoD.ravel(), bins=80, range=(0.0, 0.8), color='#1f77b4', alpha=0.85)
ax.axvline(DMAX, color='red', ls='--', lw=1.2); ax.set_xlabel('DoD'); ax.set_ylabel('count')
ax.set_title('DoD histogram (new params, 1 seed)', fontsize=10)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'dod_hist_newparams.png'), dpi=170); plt.close()
print('[done] figures ->', OUT, flush=True)
