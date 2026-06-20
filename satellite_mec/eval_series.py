"""逐槽序列捕获评估（N=25, 5400 槽, 8 策略）→ 供累计HL/系统能耗/系统时延曲线。
论文口径：满意度为主；N=192 由总量×(192/25) 投影（已验证 N-不变性）。"""
import json, time, numpy as np
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, GreedyDelayPolicy, LyapunovGreedyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.td3_sched import TD3SchedPolicy
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH=4.0; LAMBDA=4.0*Config.LAMBDA_HIGH_RATIO+Config.LAMBDA_LOW*(1-Config.LAMBDA_HIGH_RATIO)
cfg=_C(); cfg.N_EVAL_RUNS=1
env=SatelliteMECEnv(cfg)

def cap(pol,label):
    env.reset(phase='eval',seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    E=[]; D=[]; H=[]; t0=time.time()
    for _ in range(cfg.T_EVAL):
        _,_,_,info=env.step(policy=pol)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays'])); H.append(info['avg_health_loss'])
    sat=float(info.get('eval_satisfaction_rate',float('nan')))
    print(f'{label:15s} Sat={sat:.4f} ΣE={np.sum(E)/1e3:.1f}kJ ΣD={np.sum(D):.0f}s cumHL={np.sum(H):.3e} [{time.time()-t0:.0f}s]',flush=True)
    return {'satisfaction':sat,'energy':E,'delay':D,'hl':H}

out={}
out['LocalOnly']=cap(LocalOnlyPolicy(cfg,env),'LocalOnly')
out['GreedyDelay']=cap(GreedyDelayPolicy(cfg,env),'GreedyDelay')
out['LyapunovGreedy']=cap(LyapunovGreedyPolicy(cfg,env),'LyapunovGreedy')
out['GDCO']=cap(GDCOPolicy(cfg,env),'GDCO')
mh=MHSPOPolicy(cfg,env,rho_d=1.0,rho_e=1.0,V_lyapunov=10.0)
env.reset(phase='train')
for _ in range(cfg.T_WARMUP): env.step(policy=mh)
out['MHSPO']=cap(mh,'MHSPO')
td3=TD3SchedPolicy(cfg,env); td3.load('checkpoints/TD3Sched_lh4_16K'); out['TD3Sched']=cap(td3,'TD3Sched')
nobat=MAPPOPolicy(cfg,name='MAPPO_NoBat'); nobat.load('checkpoints/MAPPO_NoBat_lh4_8K'); out['MAPPO_NoBat']=cap(nobat,'MAPPO_NoBat')
lya=MAPPOPolicy(cfg,name='LyaMAPPO'); lya.load('checkpoints/LyaMAPPO_lh4_32K'); out['LyaMAPPO']=cap(lya,'LyaMAPPO')

json.dump(out, open('docs/series_lh4_n25.json','w'))
print('DONE -> docs/series_lh4_n25.json',flush=True)
