"""Multi-seed evaluation (eval-only, no retraining) — closes Codex #1 (n=1 -> CI).
Runs each policy over N_SEEDS independent eval seeds (cfg.get_eval_seeds(i)),
records per-seed headline scalars, and reports mean +/- std.
Same scenario/warmup as eval_series.py. -> docs/multiseed_lh4.json
"""
import json, time, sys, numpy as np
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, LyapunovGreedyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.td3_sched import TD3SchedPolicy
from training import MAPPOPolicy

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)

out = {}
def one_seed(pol, seed_idx):
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(seed_idx)); pol.set_eval_mode()
    H = []; Q = []; dod = []; E = 0.0; alld = []
    for _ in range(cfg.T_EVAL):
        _, _, _, info = env.step(policy=pol)
        E += info['slot_system_energy']; H.append(info['avg_health_loss'])
        Q.append(info['queue_task_count']); dod.append(info['avg_dod'])
        alld.extend(info['slot_e2e_delays'])
    return dict(
        satisfaction=float(info.get('eval_satisfaction_rate', float('nan'))),
        delay=float(np.mean(alld)) if alld else 0.0,
        hl=float(np.mean(H)), cumHL=float(np.sum(H)),
        energy_kJ=E / 1e3, queue_tasks=float(np.mean(Q)), dod=float(np.mean(dod)),
    )

def run(pol, label, warmup=False):
    if warmup:
        env.reset(phase='train')
        for _ in range(cfg.T_WARMUP): env.step(policy=pol)
    t0 = time.time(); seeds = []
    for i in range(N_SEEDS):
        seeds.append(one_seed(pol, i))
    agg = {}
    for k in seeds[0]:
        v = np.array([s[k] for s in seeds], float)
        agg[k + '_mean'] = float(v.mean()); agg[k + '_std'] = float(v.std(ddof=1) if len(v) > 1 else 0.0)
    agg['per_seed'] = seeds; agg['n_seeds'] = N_SEEDS
    out[label] = agg
    print(f"{label:15s} Sat={agg['satisfaction_mean']:.3f}±{agg['satisfaction_std']:.3f} "
          f"Delay={agg['delay_mean']:.2f}±{agg['delay_std']:.2f} "
          f"cumHL={agg['cumHL_mean']:.3f}±{agg['cumHL_std']:.3f} "
          f"E={agg['energy_kJ_mean']:.0f}±{agg['energy_kJ_std']:.0f}kJ "
          f"Q={agg['queue_tasks_mean']:.2f}±{agg['queue_tasks_std']:.2f} "
          f"[{time.time()-t0:.0f}s, n={N_SEEDS}]", flush=True)

run(LocalOnlyPolicy(cfg, env), 'LocalOnly')
run(LyapunovGreedyPolicy(cfg, env), 'LyapunovGreedy')
run(GDCOPolicy(cfg, env), 'GDCO')
run(MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0), 'MHSPO', warmup=True)
td3 = TD3SchedPolicy(cfg, env); td3.load('checkpoints/TD3Sched_lh4_16K'); run(td3, 'TD3Sched')
nb = MAPPOPolicy(cfg, name='MAPPO_NoBat'); nb.load('checkpoints/MAPPO_NoBat_lh4_8K'); run(nb, 'MAPPO_NoBat')
ly = MAPPOPolicy(cfg, name='LyaMAPPO'); ly.load('checkpoints/LyaMAPPO_lh4_32K'); run(ly, 'LyaMAPPO')

json.dump(out, open('docs/multiseed_lh4.json', 'w'), indent=2)
print(f'DONE (n={N_SEEDS}) -> docs/multiseed_lh4.json', flush=True)
