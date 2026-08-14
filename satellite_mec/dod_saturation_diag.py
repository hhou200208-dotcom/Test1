"""Battery DoD saturation diagnosis on gold BLA-MAPPO (READ-ONLY).

No core edits, no retrain, no param change. Loads checkpoints/LyaMAPPO_lh4_32K,
evaluates on 10 eval seeds x 5400 slots (lambda_high=4). Per-sat per-slot logs:
  DoD, f_cmp, comp_power(=kappa*f^3), solar_power, net delta (pre-clip drift).
net delta is reconstructed externally from stored slot energies + tracked
dod_before, and self-validated against the realized clipped DoD update.

Outputs to docs/dod_diag/:
  raw/seed{i}.npz            per-seed arrays (resumable cache)
  dod_diag_per_seed.csv      per-seed key stats
  dod_diag_aggregate.csv     pooled/all-seed metrics (incl high/low split, L')
  dod_diag_per_sat.csv       per-sat stats (touch-0.8, longest run, group)
  fig_A_dod_timeseries.(png|pdf), fig_B_dod_ecdf.(png|pdf),
  fig_C_fcmp_hist.(png|pdf),      fig_D_comppower_hist.(png|pdf)
"""
import csv, os, time
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


cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)
NS   = cfg.N_SATS          # 25 (physical satellites; DoD is per-sat/intensive -> no 192 projection)
T    = cfg.T_EVAL          # 5400
TAU  = cfg.TAU; ECAP = cfg.E_CAP
DMAX = cfg.DOD_MAX; DMIN = cfg.DOD_MIN
FMAX = cfg.CPU_FREQ; A = cfg.A_COEF
TOL  = 1e-6
NSEED = 10
CKPT = 'checkpoints/LyaMAPPO_lh4_32K'
OUT  = 'docs/dod_diag'; RAW = os.path.join(OUT, 'raw')
os.makedirs(RAW, exist_ok=True)

HIGH = np.array(sorted(env.constellation.high_load_sats))
LOW  = np.array(sorted(env.constellation.low_load_sats))
GROUP = np.array(['high' if n in set(HIGH.tolist()) else 'low' for n in range(NS)])

pol = MAPPOPolicy(cfg, name='LyaMAPPO'); pol.load(CKPT)


def lprime(d):
    """L'(delta) = 10^{a(d-1)} * (1 + a*ln10*d)  (matches lyapunov.health_loss_deriv)."""
    return (10.0 ** (A * (d - 1.0))) * (1.0 + A * LN10 * d)


def run_seed(i):
    f = os.path.join(RAW, f'seed{i}.npz')
    if os.path.exists(f):
        print(f'  seed{i}: cached, skip', flush=True); return
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(i)); pol.set_eval_mode()
    sats = env.constellation.satellites
    prev = np.array([s.dod for s in sats], float)          # dod_before at t=0
    DoD = np.empty((T, NS), np.float32); F = np.empty((T, NS), np.float32)
    CP  = np.empty((T, NS), np.float32); SOL = np.empty((T, NS), np.float32)
    DLT = np.empty((T, NS), np.float32)
    maxerr = 0.0; t0 = time.time()
    for t in range(T):
        env.step(policy=pol)
        dod = np.array([s.dod for s in sats], float)
        fcmp = np.array([s.last_cpu_freq for s in sats], float)
        ce = np.array([getattr(s, 'slot_comp_energy', 0.0)  for s in sats], float)
        te = np.array([getattr(s, 'slot_trans_energy', 0.0) for s in sats], float)
        he = np.array([getattr(s, 'slot_house_energy', 0.0) for s in sats], float)
        sol = np.array([s.solar_power for s in sats], float)
        dsolar = np.minimum(sol * TAU / ECAP, np.maximum(prev - DMIN, 0.0))
        draw = (ce + te + he) / ECAP - dsolar               # net pre-clip DoD drift
        # self-validation: clip(dod_before + draw) must equal realized dod
        maxerr = max(maxerr, float(np.abs(np.clip(prev + draw, DMIN, DMAX) - dod).max()))
        DoD[t] = dod; F[t] = fcmp; CP[t] = ce / TAU; SOL[t] = sol; DLT[t] = draw
        prev = dod
    np.savez_compressed(f, DoD=DoD, F=F, CP=CP, SOL=SOL, DLT=DLT)
    print(f'  seed{i}: done [{time.time()-t0:.0f}s]  reconstruction max|err|={maxerr:.2e}', flush=True)


print(f'[diag] high-load sats ({len(HIGH)}): {HIGH.tolist()}', flush=True)
print(f'[diag] low-load  sats ({len(LOW)}): {LOW.tolist()}', flush=True)
print(f'[diag] running {NSEED} seeds x {T} slots ...', flush=True)
for i in range(NSEED):
    run_seed(i)

# ── load all seeds ──
DoD = np.stack([np.load(os.path.join(RAW, f'seed{i}.npz'))['DoD'] for i in range(NSEED)])  # (S,T,NS)
F   = np.stack([np.load(os.path.join(RAW, f'seed{i}.npz'))['F']   for i in range(NSEED)])
CP  = np.stack([np.load(os.path.join(RAW, f'seed{i}.npz'))['CP']  for i in range(NSEED)])
SOL = np.stack([np.load(os.path.join(RAW, f'seed{i}.npz'))['SOL'] for i in range(NSEED)])
DLT = np.stack([np.load(os.path.join(RAW, f'seed{i}.npz'))['DLT'] for i in range(NSEED)])
print(f'[diag] loaded arrays {DoD.shape} (seed,slot,sat)', flush=True)

P = [10, 50, 90, 95, 99]
at_hi = np.abs(DoD - DMAX) <= TOL
at_lo = np.abs(DoD - DMIN) <= TOL


def pcts(a, ps):
    a = np.asarray(a).ravel()
    return {f'P{p}': float(np.percentile(a, p)) for p in ps}


def longest_run(mask1d):
    best = cur = 0
    for v in mask1d:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


# ── per-seed stats ──
per_seed_rows = []
for s in range(NSEED):
    d = DoD[s]; f = F[s]; cp = CP[s]
    row = {'seed': s,
           'dod_at0.8_frac': float(at_hi[s].mean()),
           'dod_at0.1_frac': float(at_lo[s].mean()),
           'dod_mean': float(d.mean()), 'dod_std': float(d.std()),
           'dod_max': float(d.max())}
    for k, v in pcts(d, P).items():
        row[f'dod_{k}'] = v
    row['fcmp_mean'] = float(f.mean()); row['fcmp_max'] = float(f.max())
    for k, v in pcts(f, [50, 90, 95, 99]).items():
        row[f'fcmp_{k}'] = v
    row['fcmp_ge0.95cn_frac'] = float((f >= 0.95 * FMAX).mean())
    row['comp_power_mean'] = float(cp.mean()); row['comp_power_max'] = float(cp.max())
    for k, v in pcts(cp, [90, 95]).items():
        row[f'comp_power_{k}'] = v
    per_seed_rows.append(row)

# ── per-sat stats (over seeds) ──
per_sat_rows = []
touch_fr = np.zeros((NSEED, NS)); run_len = np.zeros((NSEED, NS))
for s in range(NSEED):
    for n in range(NS):
        touch_fr[s, n] = at_hi[s, :, n].mean()
        run_len[s, n] = longest_run(at_hi[s, :, n])
for n in range(NS):
    per_sat_rows.append({
        'sat': n, 'group': GROUP[n],
        'dod_mean': float(DoD[:, :, n].mean()),
        'touch0.8_frac_mean': float(touch_fr[:, n].mean()),
        'touch0.1_frac_mean': float(at_lo[:, :, n].mean()),
        'longest_run0.8_mean': float(run_len[:, n].mean()),
        'longest_run0.8_max': float(run_len[:, n].max()),
    })

# ── aggregate (pooled all seeds) ──
agg = []
def add(metric, value): agg.append({'metric': metric, 'value': value})

add('pool_dod_at0.8_frac', float(at_hi.mean()))
add('pool_dod_at0.1_frac', float(at_lo.mean()))
add('pool_dod_mean', float(DoD.mean())); add('pool_dod_std', float(DoD.std()))
add('pool_dod_max', float(DoD.max()))
for k, v in pcts(DoD, P).items():
    add(f'pool_dod_{k}', v)
# per-star touch-0.8 fraction distribution (over seed x sat = 250 values)
tf = touch_fr.ravel()
add('perstar_touch0.8_mean', float(tf.mean())); add('perstar_touch0.8_median', float(np.median(tf)))
add('perstar_touch0.8_P90', float(np.percentile(tf, 90))); add('perstar_touch0.8_max', float(tf.max()))
# longest consecutive run at 0.8
rl = run_len.ravel()
add('run0.8_global_max_slots', float(rl.max()))
add('run0.8_perstar_mean', float(rl.mean())); add('run0.8_perstar_median', float(np.median(rl)))
add('run0.8_perstar_P90', float(np.percentile(rl, 90))); add('run0.8_perstar_max', float(rl.max()))
# fcmp / comp power pooled
add('pool_fcmp_mean', float(F.mean())); add('pool_fcmp_max', float(F.max()))
for k, v in pcts(F, [50, 90, 95, 99]).items():
    add(f'pool_fcmp_{k}', v)
add('pool_fcmp_ge0.95cn_frac', float((F >= 0.95 * FMAX).mean()))
add('pool_comp_power_mean', float(CP.mean())); add('pool_comp_power_max', float(CP.max()))
for k, v in pcts(CP, [90, 95]).items():
    add(f'pool_comp_power_{k}', v)
# high vs low load boundary touch (step 5)
for name, idx in [('high', HIGH), ('low', LOW)]:
    dg = DoD[:, :, idx]
    add(f'{name}_dod_mean', float(dg.mean()))
    add(f'{name}_dod_at0.8_frac', float((np.abs(dg - DMAX) <= TOL).mean()))
    add(f'{name}_dod_at0.1_frac', float((np.abs(dg - DMIN) <= TOL).mean()))
# net delta drift (pre-clip) context
add('pool_delta_mean', float(DLT.mean()))
add('high_delta_mean', float(DLT[:, :, HIGH].mean()))
add('low_delta_mean', float(DLT[:, :, LOW].mean()))
# step 8: L'(DoD) on visited samples
lp = lprime(DoD)
add('lprime_at_dodmin(0.1)', float(lprime(DMIN))); add('lprime_at_dodmax(0.8)', float(lprime(DMAX)))
add('lprime_visited_mean', float(lp.mean())); add('lprime_visited_std', float(lp.std()))
add('lprime_visited_min', float(lp.min())); add('lprime_visited_max', float(lp.max()))
add('lprime_visited_P10', float(np.percentile(lp, 10))); add('lprime_visited_P90', float(np.percentile(lp, 90)))
add('lprime_visited_cv(std/mean)', float(lp.std() / max(lp.mean(), 1e-12)))
add('lprime_high_mean', float(lprime(DoD[:, :, HIGH]).mean()))
add('lprime_low_mean', float(lprime(DoD[:, :, LOW]).mean()))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames); w.writeheader()
        for r in rows:
            w.writerow(r)


write_csv(os.path.join(OUT, 'dod_diag_per_seed.csv'), per_seed_rows)
write_csv(os.path.join(OUT, 'dod_diag_per_sat.csv'), per_sat_rows)
write_csv(os.path.join(OUT, 'dod_diag_aggregate.csv'), agg, ['metric', 'value'])
print('[diag] CSVs written', flush=True)

# ── Figure A: constellation DoD over time (median, P10-P90 band, P95, 0.8 line) ──
pooled_slot = DoD.transpose(1, 0, 2).reshape(T, NSEED * NS)   # (T, 250)
med = np.median(pooled_slot, 1); p10 = np.percentile(pooled_slot, 10, 1)
p90 = np.percentile(pooled_slot, 90, 1); p95 = np.percentile(pooled_slot, 95, 1)
x = np.arange(T)
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(x, p10, p90, color='#1f77b4', alpha=0.18, lw=0, label='P10–P90')
ax.plot(x, med, color='#1f77b4', lw=1.8, label='Median')
ax.plot(x, p95, color='#ff7f0e', lw=1.2, ls='-', label='P95')
ax.axhline(DMAX, color='red', ls='--', lw=1.2, label='DoD upper bound (0.8)')
ax.axhline(DMIN, color='gray', ls=':', lw=1.0, label='DoD lower bound (0.1)')
ax.set_xlabel('Time slot'); ax.set_ylabel('DoD'); ax.set_ylim(0.0, 0.85)
ax.legend(fontsize=8, ncol=2, loc='center right'); ax.grid(True, ls=':', alpha=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'fig_A_dod_timeseries.{ext}'), dpi=180)
plt.close()

# bonus: high vs low median overlay
fig, ax = plt.subplots(figsize=(8, 4.6))
for name, idx, c in [('high-load', HIGH, '#d62728'), ('low-load', LOW, '#2ca02c')]:
    ps = DoD[:, :, idx].transpose(1, 0, 2).reshape(T, -1)
    ax.plot(x, np.median(ps, 1), color=c, lw=1.6, label=f'{name} median')
    ax.fill_between(x, np.percentile(ps, 10, 1), np.percentile(ps, 90, 1), color=c, alpha=0.12, lw=0)
ax.axhline(DMAX, color='red', ls='--', lw=1.2, label='0.8'); ax.axhline(DMIN, color='gray', ls=':', lw=1.0, label='0.1')
ax.set_xlabel('Time slot'); ax.set_ylabel('DoD'); ax.set_ylim(0.0, 0.85)
ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_A2_dod_high_vs_low.png'), dpi=180); plt.close()

# ── Figure B: ECDF + histogram of all sat-slot DoD (0.8 pile-up) ──
flat = DoD.ravel()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
xs = np.sort(flat[::37])                                       # subsample for ECDF speed
a1.plot(xs, np.linspace(0, 1, len(xs)), color='#1f77b4', lw=1.8)
a1.axvline(DMAX, color='red', ls='--', lw=1.2); a1.axvline(DMIN, color='gray', ls=':', lw=1.0)
a1.set_xlabel('DoD'); a1.set_ylabel('ECDF'); a1.set_title('DoD ECDF (all sat-slots)')
a1.grid(True, ls=':', alpha=0.4)
a2.hist(flat, bins=80, range=(0.1, 0.8), color='#1f77b4', alpha=0.85)
a2.axvline(DMAX, color='red', ls='--', lw=1.2); a2.set_xlabel('DoD'); a2.set_ylabel('count')
a2.set_title('DoD histogram (all sat-slots)')
for a in (a1, a2):
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'fig_B_dod_ecdf.{ext}'), dpi=180)
plt.close()

# ── Figure C/D: f_cmp and comp power distributions (optional) ──
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist((F / 1e9).ravel(), bins=80, color='#9467bd', alpha=0.85)
ax.axvline(0.95 * FMAX / 1e9, color='red', ls='--', lw=1.2, label='0.95·c_n')
ax.axvline(FMAX / 1e9, color='k', ls=':', lw=1.0, label='c_n=2 GHz')
ax.set_xlabel('f_cmp (GHz)'); ax.set_ylabel('count'); ax.legend(fontsize=8)
ax.grid(True, ls=':', alpha=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'fig_C_fcmp_hist.{ext}'), dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist(CP.ravel(), bins=80, color='#8c564b', alpha=0.85)
ax.axvline(cfg.P_SOLAR_MAX, color='orange', ls='--', lw=1.2, label='solar peak 30 W')
ax.set_xlabel('Computing power κ·f³ (W)'); ax.set_ylabel('count'); ax.legend(fontsize=8)
ax.grid(True, ls=':', alpha=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'fig_D_comppower_hist.{ext}'), dpi=180)
plt.close()

print('[diag] figures written to', OUT, flush=True)
print('[diag] KEY:', flush=True)
print(f"   pooled DoD==0.8 frac = {at_hi.mean()*100:.2f}%  |  ==0.1 frac = {at_lo.mean()*100:.2f}%", flush=True)
print(f"   high-load 0.8 frac = {(np.abs(DoD[:,:,HIGH]-DMAX)<=TOL).mean()*100:.2f}%  "
      f"low-load 0.8 frac = {(np.abs(DoD[:,:,LOW]-DMAX)<=TOL).mean()*100:.2f}%", flush=True)
print(f"   run0.8 global max = {int(rl.max())} slots  |  L'(visited) mean={lp.mean():.3f} "
      f"min={lp.min():.3f} max={lp.max():.3f}", flush=True)
