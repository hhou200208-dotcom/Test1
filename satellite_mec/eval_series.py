"""逐槽序列捕获评估（N=25, 5400 槽, 8 策略）。
一次产出两份数据：
  - docs/series_lh4_n25.json  : 每槽 [系统能耗, 系统总时延, HL, 队列任务数]（曲线用）
  - docs/scoreboard7_lh4.json : scalar 均值 [满意度, 时延avg, HL, DoD, 队列MB, 队列任务数]（柱状/Pareto用）
论文口径：满意度为主；队列积压用任务个数；N=192 由总量×(192/25) 投影。"""
import json, time, numpy as np
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, GreedyDelayPolicy, LyapunovGreedyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.td3_sched import TD3SchedPolicy
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH=4.0; LAMBDA=4.0*Config.LAMBDA_HIGH_RATIO+Config.LAMBDA_LOW*(1-Config.LAMBDA_HIGH_RATIO)
cfg=_C(); cfg.N_EVAL_RUNS=1
env=SatelliteMECEnv(cfg)

series={}; scalar={}
def cap(pol,label):
    env.reset(phase='eval',seeds=cfg.get_eval_seeds(0)); pol.set_eval_mode()
    E=[]; D=[]; H=[]; Q=[]; SAT=[]; DEN=[]; dod=[]; qmb=[]; alld=[]; t0=time.time()
    for _ in range(cfg.T_EVAL):
        _,_,_,info=env.step(policy=pol)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
        H.append(info['avg_health_loss']);    Q.append(info['queue_task_count'])
        SAT.append(info['slot_satisfaction_rate']); DEN.append(info['done_tasks']+info['slot_timeout'])
        dod.append(info['avg_dod']);          qmb.append(info['total_queue_size']/1e6)
        alld.extend(info['slot_e2e_delays'])
    sat=float(info.get('eval_satisfaction_rate',float('nan')))
    series[label]={'satisfaction':sat,'energy':E,'delay':D,'hl':H,'queue_tasks':Q,
                   'sat_slot':SAT,'sat_denom':DEN}
    scalar[label]={'satisfaction':sat,'delay':float(np.mean(alld)) if alld else 0.0,
                   'hl':float(np.mean(H)),'dod':float(np.mean(dod)),
                   'queue_mb':float(np.mean(qmb)),'queue_tasks':float(np.mean(Q)),
                   'cr_internal':float(env.get_eval_completion_rate())}
    print(f'{label:15s} Sat={sat:.4f} Qtasks={np.mean(Q):.2f} QMB={np.mean(qmb):.1f} '
          f'ΣE={np.sum(E)/1e3:.0f}kJ cumHL={np.sum(H):.3e} [{time.time()-t0:.0f}s]',flush=True)

cap(LocalOnlyPolicy(cfg,env),'LocalOnly')
cap(GreedyDelayPolicy(cfg,env),'GreedyDelay')
cap(LyapunovGreedyPolicy(cfg,env),'LyapunovGreedy')
cap(GDCOPolicy(cfg,env),'GDCO')
mh=MHSPOPolicy(cfg,env,rho_d=1.0,rho_e=1.0,V_lyapunov=10.0)
env.reset(phase='train')
for _ in range(cfg.T_WARMUP): env.step(policy=mh)
cap(mh,'MHSPO')
td3=TD3SchedPolicy(cfg,env); td3.load('checkpoints/TD3Sched_lh4_16K'); cap(td3,'TD3Sched')
nobat=MAPPOPolicy(cfg,name='MAPPO_NoBat'); nobat.load('checkpoints/MAPPO_NoBat_lh4_8K'); cap(nobat,'MAPPO_NoBat')
lya=MAPPOPolicy(cfg,name='LyaMAPPO'); lya.load('checkpoints/LyaMAPPO_lh4_32K'); cap(lya,'LyaMAPPO')

json.dump(series, open('docs/series_lh4_n25.json','w'))
json.dump(scalar, open('docs/scoreboard7_lh4.json','w'), indent=2)
print('DONE -> docs/series_lh4_n25.json + docs/scoreboard7_lh4.json',flush=True)
