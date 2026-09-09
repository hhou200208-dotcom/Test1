"""Real SatelliteMECEnv integration smoke test for MAPPO/IPPO/MADDPG/QMIX.
Uses the fair-comparison adapters from backbone.compare_algorithms, including
micro-step QMIX. This is still a short smoke test, not a convergence result.
"""
from __future__ import annotations
import json, math, os
import numpy as np
from core import Config, SatelliteMECEnv
from backbone.compare_algorithms import (
    SharedRewardMAPPO, IPPOPolicy, SharedRewardMADDPG,
    MicroStepQMIXPolicy, seed_everything, module_finite, system_reward,
)

SEED=17
TRAIN_SLOTS=int(os.environ.get('SMOKE_TRAIN_SLOTS','64'))
EVAL_SLOTS=int(os.environ.get('SMOKE_EVAL_SLOTS','16'))
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results','backbone_compare','real_env_smoke')

class SmokeConfig(Config):
    T_TRAIN=TRAIN_SLOTS; T_WARMUP=0; T_EVAL=EVAL_SLOTS; T_TOTAL=512
    N_EVAL_RUNS=1; K_ROLLOUT=8; EPOCH=1; MINIBATCH=64; HIDDEN_DIM=64
    BETA_TASK=0.0; BETA=0.02

def build(name,cfg,env):
    if name=='MAPPO':
        p=SharedRewardMAPPO(cfg,name='MAPPO')
    elif name=='IPPO':
        p=IPPOPolicy(cfg)
    elif name=='MADDPG':
        p=SharedRewardMADDPG(cfg,env,name='MADDPG',seed=SEED,batch_size=16,
                             start_steps=16,updates_per_slot=1,replay_size=3000,
                             reward_scale=1.0,expl_noise=.15)
    else:
        p=MicroStepQMIXPolicy(cfg,seed=SEED,batch_size=8,replay_size=1000,
                              start_transitions=8,target_interval=16,
                              eps_start=.5,eps_end=.1,eps_anneal_slots=TRAIN_SLOTS)
    p.set_train_mode(); return p

def run(name):
    seed_everything(SEED); cfg=SmokeConfig(); env=SatelliteMECEnv(cfg)
    env.reset('train',{'task':3101,'task_param':3102,'dod_init':3103})
    p=build(name,cfg,env); tr=[]; boundaries=0
    for _ in range(TRAIN_SLOTS):
        r,d,i=p.run_step(env); x=system_reward(r); assert math.isfinite(x)
        tr.append(x); boundaries+=int(d)
    if name=='QMIX': p.finalize_training()
    if name in ('MAPPO','IPPO'):
        updates=p.trainer.update_count; ok=module_finite(p.actor) and module_finite(p.critic)
    elif name=='MADDPG':
        updates=p.backbone_update_count; ok=module_finite(p.actor) and module_finite(p.critic)
    else:
        updates=p.update_count; ok=p.parameters_finite() and p.invalid_actions==0
    if updates<=0 or not ok: raise RuntimeError(f'{name}: updates={updates}, finite={ok}')
    ev=SatelliteMECEnv(cfg); ev.reset('eval',{'task':4101,'task_param':4102,'dod_init':4103})
    p.set_eval_mode(); er=[]
    for _ in range(EVAL_SLOTS):
        r,d,last=p.run_step(ev); er.append(system_reward(r))
    out={'status':'PASS','updates':int(updates),'rollout_boundaries':boundaries,
         'mean_train_Rsys':float(np.mean(tr)),'mean_eval_Rsys':float(np.mean(er)),
         'eval_completion_rate':float(ev.get_eval_completion_rate()),
         'all_params_finite':True}
    if name=='QMIX':
        out.update({'microstep_replay_size':len(p.replay),'invalid_actions':p.invalid_actions,
                    'last_loss':p.last_loss,'internal_action_dim':p.action_dim})
    return out

def main():
    os.makedirs(OUT,exist_ok=True)
    out={'stage':'real SatelliteMECEnv smoke with micro-step QMIX','seed':SEED,
         'shared_reward':'R_sys=mean(per-satellite env rewards)','beta_task':0.0,
         'qmix_adapter':'k-th task per satellite = micro-step; inactive agents use internal NOOP; within-slot discount=1; inter-slot discount=gamma'}
    for n in ('MAPPO','IPPO','MADDPG','QMIX'):
        print('\n===',n,'===',flush=True); out[n]=run(n); print(out[n],flush=True)
    path=os.path.join(OUT,'real_env_smoke_results.json')
    with open(path,'w') as f: json.dump(out,f,indent=2)
    print('ALL PASS',path)
if __name__=='__main__': main()
