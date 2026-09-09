"""Equal-budget 1K tuning runner for MAPPO, IPPO and QMIX.

MADDPG has its own three-setting sweep because it requires two optimizer rates. This
runner gives the remaining backbones the same three-candidate pilot budget before the
8K one-seed experiment.
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np

from core import SatelliteMECEnv
from backbone.compare_algorithms import (
    SharedRewardMAPPO, MicroStepQMIXPolicy, seed_everything, system_reward, module_finite,
)
from backbone.fairness_fixes import FairIPPOPolicy
from backbone.pilot_compare import PilotConfig, paired_train_seeds, fixed_eval_seeds, common_reward_from_info

ALGO=os.environ.get('TUNE_ALGO','MAPPO').upper()
TAG=os.environ.get('TUNE_TAG','default')
ACTOR_LR=float(os.environ.get('ACTOR_LR','1e-4'))
CRITIC_LR=float(os.environ.get('CRITIC_LR','1e-3'))
QMIX_LR=float(os.environ.get('QMIX_LR','5e-4'))
SEED=0; TRAIN_SLOTS=1024; EVAL_INTERVAL=256; EVAL_SLOTS=64
OUT=Path(__file__).resolve().parents[1]/'results'/'backbone_compare'/'tuning'/ALGO/TAG


def eval_policy(p,cfg):
    e=SatelliteMECEnv(cfg); e.reset('eval',fixed_eval_seeds()); p.set_eval_mode(); vals=[]
    try:
        for _ in range(EVAL_SLOTS):
            _r,_d,info=p.run_step(e); vals.append(common_reward_from_info(info,cfg.N_SATS))
    finally: p.set_train_mode()
    return {'episode_return':float(np.sum(vals)),'slot_Rsys':float(np.mean(vals)),
            'completion_rate':float(e.get_eval_completion_rate())}


def main():
    if ALGO not in ('MAPPO','IPPO','QMIX'): raise ValueError(ALGO)
    seed_everything(SEED); cfg=PilotConfig(); cfg.T_TRAIN=TRAIN_SLOTS; cfg.T_EVAL=EVAL_SLOTS
    cfg.LR_ACTOR=ACTOR_LR; cfg.LR_CRITIC=CRITIC_LR
    env=SatelliteMECEnv(cfg); env.reset('train',paired_train_seeds(SEED))
    if ALGO=='MAPPO': p=SharedRewardMAPPO(cfg,name='MAPPO')
    elif ALGO=='IPPO': p=FairIPPOPolicy(cfg,name='IPPO')
    else: p=MicroStepQMIXPolicy(cfg,seed=SEED,lr=QMIX_LR,batch_size=32,replay_size=5000,
        start_transitions=128,target_interval=200,eps_start=1.0,eps_end=0.05,eps_anneal_slots=TRAIN_SLOTS)
    p.set_train_mode(); rows=[]; train_r=[]; t0=time.time()
    for slot in range(1,TRAIN_SLOTS+1):
        r,_d,_i=p.run_step(env); rs=system_reward(r)
        if not math.isfinite(rs): raise RuntimeError('non-finite reward')
        train_r.append(rs)
        if slot==1 or slot%EVAL_INTERVAL==0 or slot==TRAIN_SLOTS:
            ev=eval_policy(p,cfg)
            updates=p.trainer.update_count if ALGO in ('MAPPO','IPPO') else p.update_count
            row={'train_slots':slot,'updates':int(updates),'mean_recent_train_Rsys':float(np.mean(train_r[-EVAL_INTERVAL:])),**ev}
            rows.append(row); print(json.dumps(row),flush=True)
    if ALGO=='QMIX': p.finalize_training(); finite=p.parameters_finite() and p.invalid_actions==0
    else: finite=module_finite(p.actor) and module_finite(p.critic)
    if not finite: raise RuntimeError('non-finite parameters or invalid action')
    post=np.asarray([x['episode_return'] for x in rows[1:]],float)
    out={'algorithm':ALGO,'tag':TAG,'actor_lr':ACTOR_LR,'critic_lr':CRITIC_LR,'qmix_lr':QMIX_LR,
         'status':'PASS','checkpoints':rows,'final_return':float(rows[-1]['episode_return']),
         'mean_postwarmup_return':float(np.mean(post)),'wall_time_sec':time.time()-t0,
         'protocol':'same 1K slots, seed0, environment/reward/evaluation as other tuning candidates'}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'tune.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('WROTE',OUT/'tune.json')
if __name__=='__main__': main()
