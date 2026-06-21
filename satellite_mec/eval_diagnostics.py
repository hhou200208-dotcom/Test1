"""Diagnostic eval — EMPIRICALLY verify result mechanisms (rigorous research).
Records per-sat DoD / freq / queue (split high-load vs low-load) + energy and HL
comp/trans split, to test:
  #1 HL decomposition: cumHL = effL'(energy-weighted marginal damage) x (E/E_cap)
     -> is low HL from less energy, or from spending energy at low DoD (z_n shaving)?
  #3 energy/load-balancing: transmission-energy fraction, freq mean/peak/spread
  #4 LSO hot-spot: DoD on high-load sats vs low-load sats (deep discharge where?)
  #8 delay/queue: per-sat queue peak/spread (does balancing cut the worst queue?)
N=25, 5400 slots, same seeds/warmup as eval_series.py. -> docs/diagnostics_lh4.json
"""
import json, time, math, numpy as np
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, LyapunovGreedyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.td3_sched import TD3SchedPolicy
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)
a, ECAP, N = cfg.A_COEF, cfg.E_CAP, cfg.N_SATS
hi = sorted(env.constellation.high_load_sats)
lo = sorted(env.constellation.low_load_sats)
Lp = lambda d: (10 ** (a * (d - 1))) * (1.0 + a * math.log(10) * d)   # marginal HL rate L'(delta)

out, series = {}, {}
def run(pol, label):
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    sats = env.constellation.satellites
    Ec = Et = HL = HLc = HLt = sumE = 0.0
    dod=[]; dodmax=[]; dhi=[]; dhimax=[]; dlo=[]; deep=[]; lpmean=[]
    fmean=[]; fmax=[]; fstd=[]; qmax=[]; qstd=[]
    fwd = arr = done = 0
    t0 = time.time()
    for _ in range(cfg.T_EVAL):
        _, _, _, info = env.step(policy=pol)
        Ec += info['slot_system_energy_comp']; Et += info['slot_system_energy_trans']
        HL += info['avg_health_loss']; HLc += info['avg_delta_l_comp']; HLt += info['avg_delta_l_trans']
        sumE += info['slot_system_energy_comp'] + info['slot_system_energy_trans']
        pd = np.asarray(info['per_sat_dod'], float)
        dod.append(pd.mean()); dodmax.append(pd.max())
        dhi.append(pd[hi].mean()); dhimax.append(pd[hi].max()); dlo.append(pd[lo].mean())
        deep.append(float((pd > 0.6).mean())); lpmean.append(float(np.mean([Lp(x) for x in pd])))
        fr = np.array([s.last_cpu_freq for s in sats])
        fmean.append(fr.mean()); fmax.append(fr.max()); fstd.append(fr.std())
        qc = np.array([len(s.forward_queue) + len(s.compute_queue) for s in sats])
        qmax.append(qc.max()); qstd.append(qc.std())
        fwd += info['forwarded']; arr += info['arrived']; done += info['done_tasks']
    sat = float(info.get('eval_satisfaction_rate', float('nan')))
    totHL = N * HL                                   # sum over all (sat,slot)
    effLp = totHL * ECAP / max(sumE, 1e-9)           # energy-weighted mean L'  (= cumHL_persat*ECAP/E_persat)
    summ = dict(
        satisfaction=sat,
        E_total_kJ=(Ec + Et) / 1e3, trans_frac=Et / max(Ec + Et, 1e-9),
        cumHL=HL, HL_trans_frac=HLt / max(HL, 1e-12),
        dod_mean=float(np.mean(dod)), dod_max=float(np.max(dodmax)),
        dod_p95max=float(np.percentile(dodmax, 95)),
        dod_hi_mean=float(np.mean(dhi)), dod_hi_max=float(np.mean(dhimax)),
        dod_lo_mean=float(np.mean(dlo)), deep_frac=float(np.mean(deep)),
        freq_mean_GHz=float(np.mean(fmean)) / 1e9, freq_max_GHz=float(np.mean(fmax)) / 1e9,
        freq_std_GHz=float(np.mean(fstd)) / 1e9,
        q_max=float(np.mean(qmax)), q_std=float(np.mean(qstd)),
        fwd_rate=fwd / max(arr, 1), done_rate=done / max(arr, 1),
        effLprime=effLp, meanLprime=float(np.mean(lpmean)),
        timing_ratio=effLp / max(np.mean(lpmean), 1e-9),   # <1: energy spent at low-DoD (good); >1: at high-DoD
    )
    out[label] = summ
    series[label] = dict(dod_hi=dhi, dod_lo=dlo, q_max=qmax)
    print(f"{label:15s} Sat={sat:.3f} E={summ['E_total_kJ']:.0f}kJ(tr{summ['trans_frac']*100:.0f}%) "
          f"cumHL={summ['cumHL']:.2e} | DoD μ{summ['dod_mean']:.3f} hi{summ['dod_hi_mean']:.3f} "
          f"lo{summ['dod_lo_mean']:.3f} max{summ['dod_max']:.2f} deep{summ['deep_frac']*100:.0f}% | "
          f"f μ{summ['freq_mean_GHz']:.2f} max{summ['freq_max_GHz']:.2f} σ{summ['freq_std_GHz']:.2f} | "
          f"qmax{summ['q_max']:.0f} σ{summ['q_std']:.1f} fwd{summ['fwd_rate']*100:.0f}% | "
          f"effL'{summ['effLprime']:.3f} meanL'{summ['meanLprime']:.3f} ratio{summ['timing_ratio']:.2f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

run(LocalOnlyPolicy(cfg, env), 'LocalOnly')
run(LyapunovGreedyPolicy(cfg, env), 'LyapunovGreedy')
run(GDCOPolicy(cfg, env), 'GDCO')
mh = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)
env.reset(phase='train')
for _ in range(cfg.T_WARMUP): env.step(policy=mh)
run(mh, 'MHSPO')
td3 = TD3SchedPolicy(cfg, env); td3.load('checkpoints/TD3Sched_lh4_16K'); run(td3, 'TD3Sched')
nb = MAPPOPolicy(cfg, name='MAPPO_NoBat'); nb.load('checkpoints/MAPPO_NoBat_lh4_8K'); run(nb, 'MAPPO_NoBat')
ly = MAPPOPolicy(cfg, name='LyaMAPPO'); ly.load('checkpoints/LyaMAPPO_lh4_32K'); run(ly, 'LyaMAPPO')

json.dump(out, open('docs/diagnostics_lh4.json', 'w'), indent=2)
json.dump(series, open('docs/diagnostics_series_lh4.json', 'w'))
print('DONE -> docs/diagnostics_lh4.json', flush=True)
