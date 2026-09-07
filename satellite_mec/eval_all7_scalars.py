"""统一 7 策略评估（1 seed, 5400 槽），落盘 scalar means 供出图。论文口径：满意度为主。"""
import json, time, numpy as np
from core import Config, SatelliteMECEnv
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy, LyapunovGreedyPolicy, MHSPOPolicy, GDCOPolicy)
from baselines.td3_sched import TD3SchedPolicy
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)

def evaluate(pol, label):
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    hl=[]; dod=[]; q=[]; delays=[]
    t0=time.time()
    for _ in range(cfg.T_EVAL):
        _,_,_,info = env.step(policy=pol)
        hl.append(info['avg_health_loss']); dod.append(info['avg_dod'])
        q.append(info['total_queue_size']); delays.extend(info['slot_e2e_delays'])
    r = {'satisfaction': float(info.get('eval_satisfaction_rate', float('nan'))),
         'cr_internal': float(env.get_eval_completion_rate()),
         'delay': float(np.mean(delays)), 'hl': float(np.mean(hl)),
         'dod': float(np.mean(dod)), 'queue_mb': float(np.mean(q)/1e6)}
    print(f"{label:15s} Sat={r['satisfaction']:.4f} Delay={r['delay']:.3f} HL={r['hl']:.3e} "
          f"DoD={r['dod']:.3f} Q={r['queue_mb']:.1f}MB (CR={r['cr_internal']:.4f}) [{time.time()-t0:.0f}s]", flush=True)
    return r

results = {}
# 非学习型
results['LocalOnly']      = evaluate(LocalOnlyPolicy(cfg, env), 'LocalOnly')
results['GreedyDelay']    = evaluate(GreedyDelayPolicy(cfg, env), 'GreedyDelay')
results['LyapunovGreedy'] = evaluate(LyapunovGreedyPolicy(cfg, env), 'LyapunovGreedy')
results['GDCO']           = evaluate(GDCOPolicy(cfg, env), 'GDCO')
# MHSPO（需预热）
mhspo = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)
env.reset(phase='train')
for _ in range(cfg.T_WARMUP): env.step(policy=mhspo)
results['MHSPO']          = evaluate(mhspo, 'MHSPO')
# 学习型（载 checkpoint）
td3 = TD3SchedPolicy(cfg, env); td3.load('checkpoints/TD3Sched_lh4_16K')
results['TD3Sched']       = evaluate(td3, 'TD3Sched')
mappo = MAPPOPolicy(cfg, name='LyaMAPPO'); mappo.load('checkpoints/LyaMAPPO_lh4_32K')
results['LyaMAPPO']       = evaluate(mappo, 'LyaMAPPO')

json.dump(results, open('/tmp/scoreboard7.json','w'), indent=2)
print('DONE -> /tmp/scoreboard7.json', flush=True)
