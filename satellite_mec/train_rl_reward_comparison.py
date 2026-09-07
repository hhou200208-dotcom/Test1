#!/usr/bin/env python3
"""Fresh, fixed-seed common-reward convergence experiment."""
import argparse, copy, hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from core.config import Config
from core.env import SatelliteMECEnv
from core.common_reward import COMPONENTS, REWARD_DEFINITION, config_from_env
from training.policy import MAPPOPolicy
from training.ippo import IPPOPolicy
from baselines.maddpg_dod import MADDPGDoDPolicy
from baselines.td3_sched import TD3SchedPolicy

ALGORITHMS=("MAPPO","IPPO","MADDPG","TD3")
def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8"); os.replace(tmp,path)
def cfg(seed):
    c=Config(); c.SEED=seed; c.T_TRAIN=32000; c.K_ROLLOUT=64
    c.SEED_TASK=seed; c.SEED_TASK_PARAM=seed+1; c.SEED_LINK=seed+2; c.SEED_NET=seed+3; c.SEED_TRAIN=seed+4
    return c
def make_policy(name,c,e):
    torch.manual_seed(c.SEED_NET); np.random.seed(c.SEED_TRAIN); random.seed(c.SEED_TRAIN)
    if name=="MAPPO": return MAPPOPolicy(c,name=name)
    if name=="IPPO": return IPPOPolicy(c,name=name)
    kw=dict(batch_size=256,start_steps=256,reward_scale=1.0,seed=c.SEED_NET,name=name)
    if name=="MADDPG": return MADDPGDoDPolicy(c,e,zhong_reward=False,**kw)
    return TD3SchedPolicy(c,e,**kw)
def updates(p):
    if hasattr(p,"trainer"): return {"ppo_updates":p.trainer.update_count}
    n=getattr(p,"_total_it",getattr(p,"update_count",0))
    return {"critic_updates":n,"actor_updates": n if p.name=="MADDPG" else n//p.policy_freq}
def state_digest(p):
    h=hashlib.sha256()
    for key in ("actor","critic","actor_target","critic_target"):
        if hasattr(p,key):
            for v in getattr(p,key).state_dict().values(): h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def evaluate(name,p,c,slots,seed=1042):
    e=SatelliteMECEnv(copy.deepcopy(c)); e.reset("eval",{"task":seed,"task_param":seed+1,"dod_init":seed+2})
    before=state_digest(p); replay=(getattr(getattr(p,"replay",None),"size",None),getattr(getattr(p,"replay",None),"ptr",None)); up=updates(p)
    p.set_eval_mode(); total=0.; ledger={k:0. for k in (*COMPONENTS,"total")}; delays=[]; hls=[]; queues=[]; last={}
    with torch.no_grad():
        for _ in range(slots):
            rewards,_,info=p.run_step(e); total+=sum(rewards.values()); last=info
            for k in ledger: ledger[k]+=info["reward_ledger"][k]
            delays.extend(info["slot_e2e_delays"]); hls.append(info["avg_health_loss"]); queues.append(info["queue_task_count"])
    p.set_train_mode()
    assert before==state_digest(p) and up==updates(p)
    assert replay==(getattr(getattr(p,"replay",None),"size",None),getattr(getattr(p,"replay",None),"ptr",None))
    assert abs(total-ledger["total"]) <= 1e-6*max(1,abs(total))
    return dict(eval_return_total=total,eval_return_per_slot=total/slots,completion_rate=e.get_eval_completion_rate(),
      satisfaction=e.eval_satisfied/max(e.eval_satisfaction_denom,1),avg_delay=float(np.mean(delays)) if delays else 0.,
      avg_health_loss=float(np.mean(hls)),queue_tasks_per_sat=float(np.mean(queues)),reward_ledger=ledger)
def empty_history(name,seed):
    keys="environment_steps wall_time_seconds train_return train_return_interval_mean eval_return eval_return_total eval_return_per_slot completion_rate satisfaction avg_delay avg_health_loss queue_tasks_per_sat".split()
    h={"algorithm":name,"seed":seed,"reward_definition":REWARD_DEFINITION,"train_return_definition":"mean raw system reward per environment slot since previous evaluation"}
    h.update({k:[] for k in keys}); h["gradient_updates"]=[]; h["reward_ledger"]={k:[] for k in (*COMPONENTS,"total")}; return h
def save_checkpoint(p,path):
    p.save(str(path),{"reward_definition":REWARD_DEFINITION,"checkpoint_kind":"full training state where supported"})
def train_one(name,args,root):
    c=cfg(args.seed); e=SatelliteMECEnv(c); e.reset("train",{"task":c.SEED_TASK,"task_param":c.SEED_TASK_PARAM,"dod_init":c.SEED+2}); p=make_policy(name,c,e)
    out=root/name.lower(); out.mkdir(parents=True,exist_ok=True); h=empty_history(name,args.seed); start=time.perf_counter(); interval=[]
    points=set(range(0,args.steps+1,args.eval_interval)); points.add(args.steps)
    for step in range(args.steps+1):
        if step in points:
            m=evaluate(name,p,c,args.eval_slots,args.eval_seed); mean=float(np.mean(interval)) if interval else 0.; interval=[]
            h["environment_steps"].append(step); h["gradient_updates"].append(updates(p)); h["wall_time_seconds"].append(time.perf_counter()-start)
            h["train_return"].append(mean); h["train_return_interval_mean"].append(mean)
            for k in ("eval_return_total","eval_return_per_slot","completion_rate","satisfaction","avg_delay","avg_health_loss","queue_tasks_per_sat"): h[k].append(m[k])
            h["eval_return"].append(m["eval_return_total"])
            for k in h["reward_ledger"]: h["reward_ledger"][k].append(m["reward_ledger"][k])
            atomic_json(out/"history.json",h); save_checkpoint(p,out/"checkpoint")
        if step==args.steps: break
        rewards,_,info=p.run_step(e); interval.append(sum(rewards.values()))
    return h,p,c
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--algorithms",nargs="+",default=list(ALGORITHMS),choices=ALGORITHMS); ap.add_argument("--resume",action="store_true")
    ap.add_argument("--steps",type=int,default=32000); ap.add_argument("--eval-interval",type=int,default=2000); ap.add_argument("--eval-slots",type=int,default=500); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--eval-seed",type=int,default=1042); ap.add_argument("--output",default="results/rl_reward_comparison/seed_42"); a=ap.parse_args()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True); results={}
    for name in a.algorithms:
        old=root/name.lower()/"history.json"
        if a.resume and old.exists() and json.loads(old.read_text())["environment_steps"][-1] >= a.steps:
            results[name]=json.loads(old.read_text()); continue
        results[name]=train_one(name,a,root)[0]
    c=cfg(a.seed)
    import scipy, matplotlib
    config_values={k:getattr(c,k) for k in dir(c) if k.isupper() and isinstance(getattr(c,k),(str,int,float,bool))}
    manifest={"preliminary":True,"single_seed":True,"git_commit":subprocess.getoutput("git rev-parse HEAD"),"git_branch":subprocess.getoutput("git branch --show-current"),"git_dirty":bool(subprocess.getoutput("git status --porcelain")),"versions":{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},"cpu":platform.processor(),"gpu":torch.cuda.is_available(),"seed":a.seed,"derived_seeds":{"train":a.seed+4,"network":a.seed+3,"evaluation":a.eval_seed},"config":config_values,"reward_definition":REWARD_DEFINITION,"reward_weights":config_from_env(c).to_dict(),"action_mapping":{"0":"local","1":"neighbor 1","2":"neighbor 2","3":"neighbor 3","4":"neighbor 4"},"action_masking":"illegal logits/preferences set to -1e9, then categorical(train) or argmax(eval)","training_steps":a.steps,"eval_interval":a.eval_interval,"eval_slots":a.eval_slots,"algorithms":a.algorithms,"gradient_updates":{n:h['gradient_updates'][-1] for n,h in results.items()},"wall_clock_seconds":{n:h['wall_time_seconds'][-1] for n,h in results.items()},"command":" ".join(sys.argv),"resumed":a.resume,"raw_results_directory":str(root)}
    atomic_json(root/"experiment_manifest.json",manifest)
if __name__=="__main__": main()
