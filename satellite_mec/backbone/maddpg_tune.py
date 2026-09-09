"""Small, equal-budget MADDPG tuning sweep after the 1K fairness pilot.

We vary only optimizer/exploration settings. Environment, R_sys, seeds, replay reward
accounting and evaluation protocol remain fixed. The sweep is diagnostic; it does not
select or alter reported final results by hand.
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np

from core import SatelliteMECEnv
from backbone.compare_algorithms import seed_everything, system_reward, module_finite
from backbone.fairness_fixes import FairSharedRewardMADDPG
from backbone.pilot_compare import (
    PilotConfig, paired_train_seeds, fixed_eval_seeds, common_reward_from_info,
)

SEED=0
TRAIN_SLOTS=int(os.environ.get('TUNE_TRAIN_SLOTS','1024'))
EVAL_INTERVAL=256
EVAL_SLOTS=64
TAG=os.environ.get('TUNE_TAG','a_lr1e-4')
ACTOR_LR=float(os.environ.get('ACTOR_LR','1e-4'))
CRITIC_LR=float(os.environ.get('CRITIC_LR','1e-3'))
EXPL_NOISE=float(os.environ.get('EXPL_NOISE','0.1'))
OUT=Path(__file__).resolve().parents[1]/'results'/'backbone_compare'/'tuning'/'MADDPG'/TAG


def evaluate(policy,cfg):
    env=SatelliteMECEnv(cfg); env.reset('eval',fixed_eval_seeds()); policy.set_eval_mode()
    vals=[]; info={}
    try:
        for _ in range(EVAL_SLOTS):
            _r,_d,info=policy.run_step(env)
            vals.append(common_reward_from_info(info,cfg.N_SATS))
    finally:
        policy.set_train_mode()
    return {'episode_return':float(np.sum(vals)),
            'slot_Rsys':float(np.mean(vals)),
            'completion_rate':float(env.get_eval_completion_rate())}


def main():
    seed_everything(SEED); cfg=PilotConfig(); cfg.T_TRAIN=TRAIN_SLOTS; cfg.T_EVAL=EVAL_SLOTS
    env=SatelliteMECEnv(cfg); env.reset('train',paired_train_seeds(SEED))
    p=FairSharedRewardMADDPG(cfg,env,name=f'MADDPG_{TAG}',seed=SEED,
        actor_lr=ACTOR_LR,critic_lr=CRITIC_LR,gamma=cfg.GAMMA,tau=0.01,
        batch_size=128,start_steps=256,updates_per_slot=1,replay_size=100_000,
        reward_scale=1.0,tau_gs_start=1.0,tau_gs_end=0.5,
        anneal_slots=TRAIN_SLOTS,expl_noise=EXPL_NOISE,zhong_reward=False)
    p.set_train_mode(); rows=[]; train_r=[]; t0=time.time()
    for slot in range(1,TRAIN_SLOTS+1):
        r,_d,_i=p.run_step(env); rs=system_reward(r)
        if not math.isfinite(rs): raise RuntimeError('non-finite reward')
        train_r.append(rs)
        if slot==1 or slot%EVAL_INTERVAL==0 or slot==TRAIN_SLOTS:
            ev=evaluate(p,cfg); row={'train_slots':slot,'updates':p.backbone_update_count,
                'mean_recent_train_Rsys':float(np.mean(train_r[-EVAL_INTERVAL:])),**ev}
            rows.append(row); print(json.dumps(row),flush=True)
    p.finalize_training()
    if not (module_finite(p.actor) and module_finite(p.critic)): raise RuntimeError('non-finite params')
    returns=np.asarray([x['episode_return'] for x in rows[1:]],dtype=float)
    result={'tag':TAG,'actor_lr':ACTOR_LR,'critic_lr':CRITIC_LR,'expl_noise':EXPL_NOISE,
        'seed':SEED,'status':'PASS','checkpoints':rows,'final_return':float(rows[-1]['episode_return']),
        'mean_postwarmup_return':float(np.mean(returns)) if len(returns) else float(rows[-1]['episode_return']),
        'wall_time_sec':time.time()-t0,
        'protocol':'same env/seeds/R_sys; R_sys divided across per-task replay transitions; no rollout terminal'}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'tune.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('WROTE',OUT/'tune.json')
if __name__=='__main__': main()
