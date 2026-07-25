"""Focused head-to-head: MADDPG-DoD (LyDRL-DoD) vs LyaMAPPO, n seeds, full 5400-slot eval.
Reports satisfaction / DoD / HL(1e-4) / delay / queue-tasks / cumHL with mean±std.
Fast path to the headline before the full 8-policy multiseed."""
import sys, time, numpy as np
from core import Config, SatelliteMECEnv
from baselines.maddpg_dod import MADDPGDoDPolicy
from training import MAPPOPolicy

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)

def one_seed(pol, i):
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(i)); pol.set_eval_mode()
    H=[]; Q=[]; dod=[]; alld=[]
    for _ in range(cfg.T_EVAL):
        _,_,_,info = env.step(policy=pol)
        H.append(info['avg_health_loss']); Q.append(info['queue_task_count'])
        dod.append(info['avg_dod']); alld.extend(info['slot_e2e_delays'])
    return dict(sat=float(info.get('eval_satisfaction_rate', float('nan'))),
                dod=float(np.mean(dod)), hl4=float(np.mean(H))*1e4,
                cumHL=float(np.sum(H)), delay=float(np.mean(alld)) if alld else 0.0,
                q=float(np.mean(Q)))

def run(pol, label):
    t0=time.time(); rows=[one_seed(pol,i) for i in range(N_SEEDS)]
    agg={k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows], ddof=1) if N_SEEDS>1 else 0.0)
         for k in rows[0]}
    print(f"{label:12s} Sat={agg['sat'][0]:.3f}±{agg['sat'][1]:.3f} "
          f"DoD={agg['dod'][0]:.3f}±{agg['dod'][1]:.3f} "
          f"HL={agg['hl4'][0]:.2f}±{agg['hl4'][1]:.2f}e-4 "
          f"cumHL={agg['cumHL'][0]:.3f} "
          f"Delay={agg['delay'][0]:.2f}s Q={agg['q'][0]:.1f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return agg

print(f"=== MADDPG-DoD vs LyaMAPPO, n={N_SEEDS} seeds, T_EVAL={cfg.T_EVAL} ===", flush=True)
md_ckpt = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/MADDPG_DoD_lh4_32K'
print(f"(MADDPG ckpt: {md_ckpt})", flush=True)
md = MADDPGDoDPolicy(cfg, env); md.load(md_ckpt); run(md, 'LyDRL-DoD')
ly = MAPPOPolicy(cfg, name='LyaMAPPO'); ly.load('checkpoints/LyaMAPPO_lh4_32K'); run(ly, 'LyaMAPPO')
print("DONE", flush=True)
