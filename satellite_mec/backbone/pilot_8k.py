"""Tuned 8K/seed0 pilot for MAPPO, IPPO, MADDPG and QMIX.

Hyperparameters are frozen from the equal-budget 1K screening recorded in
results/backbone_compare/tuning/tuning_summary.json. This run is a longer sanity check
before the final 32K x 5 paired-seed comparison; it is not a final paper result.
"""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np

from core import Config, SatelliteMECEnv
from backbone.compare_algorithms import (
    SharedRewardMAPPO, MicroStepQMIXPolicy, seed_everything,
    system_reward, module_finite,
)
from backbone.fairness_fixes import FairIPPOPolicy, FairSharedRewardMADDPG
from backbone.pilot_compare import paired_train_seeds, fixed_eval_seeds, common_reward_from_info

ALGO=os.environ.get('ALGORITHM','MAPPO').upper()
SEED=int(os.environ.get('PILOT_SEED','0'))
TRAIN_SLOTS=int(os.environ.get('PILOT_TRAIN_SLOTS','8192'))
EVAL_INTERVAL=int(os.environ.get('PILOT_EVAL_INTERVAL','512'))
EVAL_SLOTS=int(os.environ.get('PILOT_EVAL_SLOTS','64'))
OUT_ROOT=Path(__file__).resolve().parents[1]/'results'/'backbone_compare'/'pilot_8k'

class Pilot8KConfig(Config):
    T_WARMUP=0
    N_EVAL_RUNS=1
    K_ROLLOUT=64
    BETA_TASK=0.0
    T_TOTAL=max(TRAIN_SLOTS+256,EVAL_SLOTS+256,9000)

SELECTED={
    'MAPPO': {'actor_lr':3e-4,'critic_lr':1e-3},
    'IPPO': {'actor_lr':3e-4,'critic_lr':1e-3},
    'MADDPG': {'actor_lr':1e-4,'critic_lr':3e-4,'expl_noise':0.15},
    'QMIX': {'lr':5e-4},
}

def build_policy(name,cfg,env):
    hp=SELECTED[name]
    if name=='MAPPO':
        cfg.LR_ACTOR=hp['actor_lr']; cfg.LR_CRITIC=hp['critic_lr']
        p=SharedRewardMAPPO(cfg,name='MAPPO')
    elif name=='IPPO':
        cfg.LR_ACTOR=hp['actor_lr']; cfg.LR_CRITIC=hp['critic_lr']
        p=FairIPPOPolicy(cfg,name='IPPO')
    elif name=='MADDPG':
        p=FairSharedRewardMADDPG(cfg,env,name='MADDPG',seed=SEED,
            actor_lr=hp['actor_lr'],critic_lr=hp['critic_lr'],gamma=cfg.GAMMA,tau=0.01,
            batch_size=128,start_steps=256,updates_per_slot=1,replay_size=200_000,
            reward_scale=1.0,tau_gs_start=1.0,tau_gs_end=0.5,
            anneal_slots=TRAIN_SLOTS,expl_noise=hp['expl_noise'],zhong_reward=False)
    elif name=='QMIX':
        p=MicroStepQMIXPolicy(cfg,seed=SEED,lr=hp['lr'],batch_size=32,replay_size=10000,
            start_transitions=128,target_interval=200,eps_start=1.0,eps_end=0.05,
            eps_anneal_slots=TRAIN_SLOTS)
    else: raise ValueError(name)
    p.set_train_mode(); return p

def updates(p):
    if ALGO in ('MAPPO','IPPO'): return int(p.trainer.update_count)
    if ALGO=='MADDPG': return int(p.backbone_update_count)
    return int(p.update_count)

def finite(p):
    if ALGO in ('MAPPO','IPPO'): return module_finite(p.actor) and module_finite(p.critic)
    if ALGO=='MADDPG': return module_finite(p.actor) and module_finite(p.critic)
    return p.parameters_finite() and p.invalid_actions==0

def evaluate(p,cfg):
    env=SatelliteMECEnv(cfg); env.reset('eval',fixed_eval_seeds()); p.set_eval_mode()
    vals=[]; info={}
    try:
        for _ in range(EVAL_SLOTS):
            _r,_d,info=p.run_step(env)
            vals.append(common_reward_from_info(info,cfg.N_SATS))
    finally: p.set_train_mode()
    return {
      'mean_eval_episode_return':float(np.sum(vals)),
      'mean_eval_slot_Rsys':float(np.mean(vals)),
      'eval_completion_rate':float(env.get_eval_completion_rate()),
      'eval_avg_dod':float(info.get('avg_dod',0.0)),
      'eval_avg_health_loss':float(info.get('avg_health_loss',0.0)),
    }

def main():
    if ALGO not in SELECTED: raise ValueError(ALGO)
    seed_everything(SEED); cfg=Pilot8KConfig(); cfg.T_TRAIN=TRAIN_SLOTS; cfg.T_EVAL=EVAL_SLOTS
    env=SatelliteMECEnv(cfg); env.reset('train',paired_train_seeds(SEED)); p=build_policy(ALGO,cfg,env)
    rows=[]; train_r=[]; t0=time.time()
    for slot in range(1,TRAIN_SLOTS+1):
        r,_d,_i=p.run_step(env); rs=system_reward(r)
        if not math.isfinite(rs): raise RuntimeError(f'{ALGO}: non-finite reward at {slot}')
        train_r.append(rs)
        if slot==1 or slot%EVAL_INTERVAL==0 or slot==TRAIN_SLOTS:
            ev=evaluate(p,cfg)
            row={'train_slots':slot,'updates':updates(p),
                 'mean_recent_train_Rsys':float(np.mean(train_r[-EVAL_INTERVAL:])),**ev}
            rows.append(row); print(json.dumps(row),flush=True)
    if ALGO in ('MADDPG','QMIX'): p.finalize_training()
    if not finite(p): raise RuntimeError(f'{ALGO}: invalid/non-finite model')
    post=np.asarray([r['mean_eval_episode_return'] for r in rows[1:]],float)
    result={'algorithm':ALGO,'seed':SEED,'status':'PASS','selected_hyperparameters':SELECTED[ALGO],
      'protocol':{'n_sats':cfg.N_SATS,'train_slots':TRAIN_SLOTS,'eval_interval_slots':EVAL_INTERVAL,
        'eval_slots':EVAL_SLOTS,'rollout_slots':cfg.K_ROLLOUT,'beta_task':0.0,'gamma':cfg.GAMMA,
        'shared_reward':'R_sys=mean_n r_n','paired_train_seeds':paired_train_seeds(SEED),
        'fixed_eval_seeds':fixed_eval_seeds(),'evaluation_reward':'full reward reconstructed from ledger'},
      'checkpoints':rows,'final_updates':updates(p),'final_return':float(rows[-1]['mean_eval_episode_return']),
      'mean_post_initial_return':float(np.mean(post)),'auc_trapezoid':float(np.trapezoid(post,dx=EVAL_INTERVAL)) if len(post)>1 else 0.0,
      'wall_time_sec':time.time()-t0,'all_params_finite':True}
    if ALGO=='QMIX': result['qmix']={'microstep':'kth task decision round + internal NOOP','replay_size':len(p.replay),'last_loss':p.last_loss,'invalid_actions':p.invalid_actions}
    out=OUT_ROOT/ALGO/f'seed{SEED}'; out.mkdir(parents=True,exist_ok=True)
    (out/'pilot_8k.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('WROTE',out/'pilot_8k.json')
if __name__=='__main__': main()
