"""Plan-甲: compare cumulative HL under the TANGENT (current) vs SECANT (proposed
counterfactual) metric on the SAME trajectories. Read-only external logging.

For each method x seed (5 methods x 5 seeds x 5400 slots) log per-sat per-slot:
  DoD (after-update), E_comp, E_trans, E_house, solar_power.
Then reconstruct, per sat-slot (dod_before = previous DoD):
  tangent : dL_tan = L'(dod_before)*(dComp+dTrans)              == code slot_health_loss
  secant  : dL_sec = L(DoD_with) - L(DoD_without)
            DoD_with    = clip(dod_before + dBase + dComp + dTrans - dSolar)   (== realized DoD)
            DoD_without = clip(dod_before + dBase - dSolar)                     (no comp/trans)
  clip0   : controllable energy spent (dComp+dTrans>0) AND DoD_without already clipped to DoD_max
            -> secant collapses to 0 (the saturation artifact we are checking)
Aggregate: cumulative over t, sum over sats, x(192/25) projection.
Outputs docs/hl_metric_compare/: per-method npz cache, CSV, comparison figure.
No retrain, no core edits, gold untouched.
"""
import csv, json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.maddpg_dod import MADDPGDoDPolicy
from training import MAPPOPolicy

LN10 = np.log(10.0)


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)


cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)
NS = cfg.N_SATS; T = cfg.T_EVAL
TAU = cfg.TAU; ECAP = cfg.E_CAP; DMIN = cfg.DOD_MIN; DMAX = cfg.DOD_MAX; A = cfg.A_COEF
NSEED = 5; RATIO = 192.0 / NS; TOL = 1e-6
OUT = 'docs/hl_metric_compare'; RAW = os.path.join(OUT, 'raw'); os.makedirs(RAW, exist_ok=True)


def Lfun(d):
    return d * (10.0 ** (A * (d - 1.0)))


def Lprime(d):
    return (10.0 ** (A * (d - 1.0))) * (1.0 + A * LN10 * d)


def run(pol, key, warmup=False):
    f = os.path.join(RAW, f'{key}.npz')
    if os.path.exists(f):
        print(f'  {key}: cached', flush=True); return
    if warmup:
        env.reset(phase='train')
        for _ in range(cfg.T_WARMUP):
            env.step(policy=pol)
    DoD = np.empty((NSEED, T, NS), np.float32); EC = np.empty_like(DoD)
    ET = np.empty_like(DoD); EH = np.empty_like(DoD); SO = np.empty_like(DoD)
    SHL = np.empty_like(DoD)                       # code's slot_health_loss (tangent ground-truth)
    dod0 = np.empty((NSEED, NS), np.float32)       # dod_before at t=0
    for s in range(NSEED):
        env.reset(phase='eval', seeds=cfg.get_eval_seeds(s)); pol.set_eval_mode()
        sats = env.constellation.satellites
        dod0[s] = np.array([x.dod for x in sats], float)
        t0 = time.time()
        for t in range(T):
            env.step(policy=pol)
            DoD[s, t] = [x.dod for x in sats]
            EC[s, t] = [getattr(x, 'slot_comp_energy', 0.0) for x in sats]
            ET[s, t] = [getattr(x, 'slot_trans_energy', 0.0) for x in sats]
            EH[s, t] = [getattr(x, 'slot_house_energy', 0.0) for x in sats]
            SO[s, t] = [x.solar_power for x in sats]
            SHL[s, t] = [getattr(x, 'slot_health_loss', 0.0) for x in sats]
        print(f'  {key} seed{s}: [{time.time()-t0:.0f}s]', flush=True)
    np.savez_compressed(f, DoD=DoD, EC=EC, ET=ET, EH=EH, SO=SO, SHL=SHL, dod0=dod0)
    print(f'== {key} DONE ==', flush=True)


METHODS = [
    ('LyaMAPPO',   lambda: _load(MAPPOPolicy(cfg, name='LyaMAPPO'), 'checkpoints/LyaMAPPO_lh4_32K'), False),
    ('MADDPG_DoD', lambda: _load(MADDPGDoDPolicy(cfg, env), 'checkpoints/MADDPG_DoD_lh4_32K'), False),
    ('MHSPO',      lambda: MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0), True),
    ('GDCO',       lambda: GDCOPolicy(cfg, env), False),
    ('LocalOnly',  lambda: LocalOnlyPolicy(cfg, env), False),
]


def _load(p, path):
    p.load(path); return p


for key, make, warm in METHODS:
    run(make(), key, warmup=warm)

# ── analysis ──
def compute(key):
    d = np.load(os.path.join(RAW, f'{key}.npz'))
    DoD, EC, ET, EH, SO, SHL, dod0 = (d['DoD'], d['EC'], d['ET'], d['EH'], d['SO'], d['SHL'], d['dod0'])
    # dod_before: shift DoD by one slot, first slot uses dod0
    before = np.empty_like(DoD)
    before[:, 0, :] = dod0
    before[:, 1:, :] = DoD[:, :-1, :]
    dComp = EC / ECAP; dTrans = ET / ECAP; dBase = EH / ECAP
    dSolar = np.minimum(SO * TAU / ECAP, np.maximum(before - DMIN, 0.0))
    dod_with = np.clip(before + dBase + dComp + dTrans - dSolar, DMIN, DMAX)
    dod_wo   = np.clip(before + dBase - dSolar, DMIN, DMAX)
    err = float(np.abs(dod_with - DoD).max())                 # must ~0 (validates reconstruction)
    dL_tan = SHL                                              # code ground-truth tangent
    dL_tan_chk = Lprime(before) * (dComp + dTrans)            # our reconstruction
    tan_err = float(np.abs(dL_tan - dL_tan_chk).max())
    dL_sec = Lfun(dod_with) - Lfun(dod_wo)
    ctrl = (dComp + dTrans) > 0
    clip0 = ctrl & (dod_wo >= DMAX - TOL)                     # controllable spent but secant zeroed by ceiling
    return dict(before=before, dL_tan=dL_tan, dL_sec=dL_sec, ctrl=ctrl, clip0=clip0,
                DoD=DoD, err=err, tan_err=tan_err)


R = {key: compute(key) for key, _, _ in METHODS}
print('[chk] reconstruction / tangent errors:',
      {k: (round(v['err'], 12), round(v['tan_err'], 12)) for k, v in R.items()}, flush=True)

DISP = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD', 'LocalOnly': 'LSO'}
disp = lambda k: DISP.get(k, k)
ORDER = ['LyaMAPPO', 'MADDPG_DoD', 'MHSPO', 'GDCO', 'LocalOnly']

rows = []
for k in ORDER:
    r = R[k]
    tan_final = float(r['dL_tan'].sum(2).mean(0).sum() * RATIO)   # sum sat, mean seed, sum t, xproj
    sec_final = float(r['dL_sec'].sum(2).mean(0).sum() * RATIO)
    ctrl_n = float(r['ctrl'].sum())
    clip0_frac = float(r['clip0'].sum() / max(ctrl_n, 1))         # among controllable-energy slots
    sat_frac = float((np.abs(r['DoD'] - DMAX) <= TOL).mean())
    rows.append({'method': disp(k), 'tangent_cumHL': tan_final, 'secant_cumHL': sec_final,
                 'secant/tangent': sec_final / max(tan_final, 1e-12),
                 'clip0_frac_of_ctrl': clip0_frac, 'dod_at0.8_frac': sat_frac})

# rankings (lower cumHL = better)
tan_rank = sorted(range(len(rows)), key=lambda i: rows[i]['tangent_cumHL'])
sec_rank = sorted(range(len(rows)), key=lambda i: rows[i]['secant_cumHL'])
for rk, i in enumerate(tan_rank):
    rows[i]['tangent_rank'] = rk + 1
for rk, i in enumerate(sec_rank):
    rows[i]['secant_rank'] = rk + 1

with open(os.path.join(OUT, 'hl_metric_compare.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['method', 'tangent_cumHL', 'tangent_rank', 'secant_cumHL',
                                       'secant_rank', 'secant/tangent', 'clip0_frac_of_ctrl',
                                       'dod_at0.8_frac'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print('\n=== HL METRIC COMPARISON (5 seeds, x192/25) ===', flush=True)
print(f"{'method':12} {'tan_cumHL':>12} {'rk':>3} {'sec_cumHL':>12} {'rk':>3} {'sec/tan':>8} {'clip0%':>7} {'sat0.8%':>8}", flush=True)
for r in rows:
    print(f"{r['method']:12} {r['tangent_cumHL']:12.4f} {r['tangent_rank']:3d} "
          f"{r['secant_cumHL']:12.4f} {r['secant_rank']:3d} {r['secant/tangent']:8.3f} "
          f"{r['clip0_frac_of_ctrl']*100:6.1f}% {r['dod_at0.8_frac']*100:7.1f}%", flush=True)
tan_order = [rows[i]['method'] for i in tan_rank]
sec_order = [rows[i]['method'] for i in sec_rank]
print('tangent ranking (best->worst):', ' < '.join(tan_order), flush=True)
print('secant  ranking (best->worst):', ' < '.join(sec_order), flush=True)
print('RANKING STABLE:', tan_order == sec_order, flush=True)

# ── figure: cumulative curves, tangent (left) vs secant (right) ──
COLOR = {'LyaMAPPO': '#d62728', 'MADDPG_DoD': '#F2C200', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for k in ORDER:
    r = R[k]; x = np.arange(T)
    big = (k == 'LyaMAPPO')
    yt = np.cumsum(r['dL_tan'].sum(2).mean(0)) * RATIO
    ys = np.cumsum(r['dL_sec'].sum(2).mean(0)) * RATIO
    a1.plot(x, yt, color=COLOR[k], lw=2.4 if big else 1.5, label=disp(k), zorder=3 if big else 2)
    a2.plot(x, ys, color=COLOR[k], lw=2.4 if big else 1.5, label=disp(k), zorder=3 if big else 2)
a1.set_title('Tangent (current):  ' + r"$\sum_t\sum_n \mathcal{L}'(DoD)\,\Delta DoD$", fontsize=11)
a2.set_title('Secant (proposed):  ' + r"$\sum_t\sum_n [\mathcal{L}(DoD_{with})-\mathcal{L}(DoD_{wo})]$", fontsize=11)
for a in (a1, a2):
    a.set_xlabel('Time slot'); a.set_ylabel('System cumulative health loss')
    a.legend(fontsize=8, ncol=2); a.grid(True, ls=':', alpha=0.4)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(OUT, f'hl_metric_compare.{ext}'), dpi=170)
plt.close()
print('[done] figure + CSV ->', OUT, flush=True)
