"""Diagnostic eval v2 — addresses Codex review (mechanism causality, exact口径).

Fixes vs v1:
  * HL "exposure/timing" factor now uses dod_BEFORE (exact, matches the code's
    HL = L'(dod_before)*dE/E_cap), via per-slot prev-DoD tracking + per-sat energy.
    Reports effLprime (energy-weighted L', exact) and meanLprime (unweighted),
    timing = effLprime/meanLprime. The claim-B HL factor is effLprime RELATIVE to
    LyaMAPPO (added in post-analysis), NOT the old effL'/meanL' ratio.
  * Energy peak-shaving made identifiable: logs per-(sat,slot) frequency and
    decomposes compute energy  E ∝ Σf³ = (N·T)·mean(f³) = (N·T)·mean(f)³·D,
    where D = mean(f³)/mean(f)³ ≥ 1 is the dispersion/peak multiplier (Jensen).
    -> separates "lower clock level (mean f)" from "less peaking (D)".
Same seeds/warmup as eval_series.py. -> docs/diagnostics_lh4.json
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
Lp = lambda d: (10 ** (a * (d - 1))) * (1.0 + a * math.log(10) * d)

out = {}
def run(pol, label):
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    sats = env.constellation.satellites
    prev_dod = np.array([s.dod for s in sats])          # dod_before for slot 0 = post-reset DoD
    Ec = Et = HL = 0.0
    sumLp = sumLpE = sumE_ps = 0.0; cntsat = 0          # for exact effL'/meanL' (dod_before)
    sumf = sumf3 = 0.0; cntf = 0                         # for energy level/dispersion decomposition
    dod=[]; dodmax=[]; dhi=[]; dhimax=[]; dlo=[]; deep=[]
    fmax=[]; fstd=[]; qmax=[]; qstd=[]
    fwd = arr = done = 0
    t0 = time.time()
    for _ in range(cfg.T_EVAL):
        dod_before = prev_dod                            # exact: dod at start of this slot
        _, _, _, info = env.step(policy=pol)
        Ec += info['slot_system_energy_comp']; Et += info['slot_system_energy_trans']
        HL += info['avg_health_loss']
        e_ps = np.array([s.slot_comp_energy + s.slot_trans_energy for s in sats])
        f_ps = np.array([s.last_cpu_freq for s in sats])
        lp_b = np.array([Lp(d) for d in dod_before])
        sumLp += lp_b.sum(); cntsat += len(lp_b)
        sumLpE += float((lp_b * e_ps).sum()); sumE_ps += float(e_ps.sum())
        sumf += float(f_ps.sum()); sumf3 += float((f_ps ** 3).sum()); cntf += len(f_ps)
        pd = np.asarray(info['per_sat_dod'], float)
        dod.append(pd.mean()); dodmax.append(pd.max())
        dhi.append(pd[hi].mean()); dhimax.append(pd[hi].max()); dlo.append(pd[lo].mean())
        deep.append(float((pd > 0.6).mean()))
        fmax.append(f_ps.max()); fstd.append(f_ps.std())
        qc = np.array([len(s.forward_queue) + len(s.compute_queue) for s in sats])
        qmax.append(qc.max()); qstd.append(qc.std())
        fwd += info['forwarded']; arr += info['arrived']; done += info['done_tasks']
        prev_dod = pd
    sat = float(info.get('eval_satisfaction_rate', float('nan')))
    effLp = sumLpE / max(sumE_ps, 1e-9)                  # energy-weighted L' (dod_before, EXACT)
    meanLp = sumLp / max(cntsat, 1)                      # unweighted L' (dod_before)
    effLp_check = (N * HL) * ECAP / max(Ec + Et, 1e-9)   # must ≈ effLp (sanity)
    mean_f = sumf / max(cntf, 1); mean_f3 = sumf3 / max(cntf, 1)
    disp = mean_f3 / max(mean_f ** 3, 1e-30)             # dispersion/peak multiplier D≥1
    out[label] = dict(
        satisfaction=sat, E_total_kJ=(Ec + Et) / 1e3, trans_frac=Et / max(Ec + Et, 1e-9),
        cumHL=HL,
        effLprime=effLp, effLprime_check=effLp_check, meanLprime=meanLp,
        timing=effLp / max(meanLp, 1e-9),               # >1: energy spent at high-DoD slots
        mean_f_GHz=mean_f / 1e9, dispersion_D=disp,     # energy = (N·T)·mean_f³·D
        sumf3=sumf3,
        dod_mean=float(np.mean(dod)), dod_max=float(np.max(dodmax)),
        dod_hi_mean=float(np.mean(dhi)), dod_hi_max=float(np.mean(dhimax)),
        dod_lo_mean=float(np.mean(dlo)), deep_frac=float(np.mean(deep)),
        freq_max_GHz=float(np.mean(fmax)) / 1e9, freq_std_GHz=float(np.mean(fstd)) / 1e9,
        q_max=float(np.mean(qmax)), q_std=float(np.mean(qstd)),
        fwd_rate=fwd / max(arr, 1), done_rate=done / max(arr, 1),
    )
    d = out[label]
    print(f"{label:15s} Sat={sat:.3f} E={d['E_total_kJ']:.0f}kJ cumHL={d['cumHL']:.2e} | "
          f"effL'{d['effLprime']:.3f}(chk{d['effLprime_check']:.3f}) meanL'{d['meanLprime']:.3f} time{d['timing']:.2f} | "
          f"f_μ{d['mean_f_GHz']:.2f} D{d['dispersion_D']:.2f} fmax{d['freq_max_GHz']:.2f} | "
          f"DoD hi{d['dod_hi_mean']:.3f} lo{d['dod_lo_mean']:.3f} | qmax{d['q_max']:.0f} fwd{d['fwd_rate']*100:.0f}% "
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
print('DONE -> docs/diagnostics_lh4.json', flush=True)
