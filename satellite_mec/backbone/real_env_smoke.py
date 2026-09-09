"""Real SatelliteMECEnv integration smoke test for MAPPO/IPPO/MADDPG/QMIX.
This is not the final convergence experiment. QMIX uses a slot-level adapter here;
formal pilot work will replace it with the planned micro-step wrapper.
"""
from __future__ import annotations
import json, math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import Config, SatelliteMECEnv
from training import MAPPOPolicy
from training.trainer import MAPPOTrainer
from baselines.maddpg_dod import MADDPGDoDPolicy
from interfaces import PolicyInterface

SEED=17
TRAIN_SLOTS=int(os.environ.get('SMOKE_TRAIN_SLOTS','32'))
EVAL_SLOTS=int(os.environ.get('SMOKE_EVAL_SLOTS','8'))
OUT=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results','backbone_compare','real_env_smoke')

def seed_all(x):
    random.seed(x); np.random.seed(x); torch.manual_seed(x)

class SmokeConfig(Config):
    T_TRAIN=TRAIN_SLOTS; T_WARMUP=0; T_EVAL=EVAL_SLOTS; T_TOTAL=256
    N_EVAL_RUNS=1; K_ROLLOUT=8; EPOCH=1; MINIBATCH=64; HIDDEN_DIM=64
    BETA_TASK=0.0; BETA=0.02

class SharedMAPPO(MAPPOPolicy):
    """Vanilla MAPPO backbone: task credit off, every critic gets R_sys."""
    def store_slot_data(self,t,rewards,n_executed,done):
        r=float(np.mean(list(rewards.values()))) if rewards else 0.0
        for n in range(self.cfg.N_SATS):
            if n in self._slot_critic:
                s,v=self._slot_critic[n]; self.buffer.add_slot(n,t,s,r,v,done)

class LocalCritic(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.d=cfg.get_state_dim()-7; h=cfg.HIDDEN_DIM
        self.norm=nn.LayerNorm(self.d)
        self.net=nn.Sequential(nn.Linear(self.d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def get_value(self,x):
        squeeze=x.dim()==1
        if squeeze: x=x.unsqueeze(0)
        y=self.net(self.norm(x)).squeeze(-1)
        return y.squeeze(0) if squeeze else y

class IPPOTrainer(MAPPOTrainer):
    def __init__(self,cfg):
        super().__init__(cfg); self.critic=LocalCritic(cfg).to(self.device)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=cfg.LR_CRITIC)
    def compute_bootstrap(self,env,buffer):
        d=self.cfg.get_state_dim()-7; out={}
        for n,s in env.get_critic_obs().items():
            x=torch.as_tensor(np.asarray(s[:d],np.float32),device=self.device)
            with torch.no_grad(): out[n]=float(self.critic.get_value(x).item())
        return out

class IPPO(SharedMAPPO):
    def __init__(self,cfg):
        super().__init__(cfg,name='IPPO'); self.trainer=IPPOTrainer(cfg)
        self.actor=self.trainer.actor; self.critic=self.trainer.critic
    def collect_critic_values(self,env):
        d=self.cfg.get_state_dim()-7; self._slot_critic={}
        for n,s in env.get_critic_obs().items():
            x=np.asarray(s[:d],np.float32); xt=torch.as_tensor(x,device=self.trainer.device)
            with torch.no_grad(): v=float(self.critic.get_value(xt).item())
            self._slot_critic[n]=(x,v)

class SharedMADDPG(MADDPGDoDPolicy):
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.smoke_updates=0
    def _flush_slot(self,rewards,done,info=None):
        if not self._slot_tasks: return
        r=float(np.mean(list(rewards.values()))) if rewards else 0.0
        for s,c,a,m,ac,n in self._slot_tasks:
            if self._pending is not None:
                ps,pc,pa,pr=self._pending; self.replay.add(ps,pc,pa,pr,s,c,m,0.0); self._env_steps+=1
            self._pending=(s,c,a,r*self.reward_scale)
        if done and self._pending is not None:
            ps,pc,pa,pr=self._pending
            self.replay.add(ps,pc,pa,pr,np.zeros(self.s_dim,np.float32),np.zeros(self.c_dim,np.float32),np.ones(self.a_dim,np.float32),1.0)
            self._env_steps+=1; self._pending=None
    def _update(self):
        ok=self.replay.size>=max(self.batch_size,self.start_steps); super()._update()
        if ok: self.smoke_updates+=1

class AgentQ(nn.Module):
    def __init__(self,cfg):
        super().__init__(); h=cfg.HIDDEN_DIM
        self.net=nn.Sequential(nn.LayerNorm(cfg.get_state_dim()),nn.Linear(cfg.get_state_dim(),h),nn.ReLU(),nn.Linear(h,cfg.get_action_dim()))
    def forward(self,x): return self.net(x)

class Mixer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); h=cfg.HIDDEN_DIM; g=cfg.get_critic_state_dim(); n=cfg.N_SATS
        self.w=nn.Sequential(nn.Linear(g,h),nn.ReLU(),nn.Linear(h,n)); self.b=nn.Linear(g,1)
    def forward(self,q,g): return (F.softplus(self.w(g))*q).sum(-1)+self.b(g).squeeze(-1)

class SmokeQMIX(PolicyInterface):
    needs_training=True; name='QMIX'
    def __init__(self,cfg):
        self.cfg=cfg; self.q=AgentQ(cfg); self.mix=Mixer(cfg); self.opt=torch.optim.Adam(list(self.q.parameters())+list(self.mix.parameters()),lr=5e-4)
        self.eval=False; self.eps=.25; self.rng=np.random.default_rng(SEED); self.cobs={}; self.tasks=[]; self.update_count=0; self.invalid=0
    def set_eval_mode(self): self.eval=True; self.q.eval(); self.mix.eval()
    def set_train_mode(self): self.eval=False; self.q.train(); self.mix.train()
    def collect_critic_values(self,env): self.cobs={n:np.asarray(v,np.float32) for n,v in env.get_critic_obs().items()}; self.tasks=[]
    def act_one(self,state,mask):
        m=np.asarray(mask,np.float32); feasible=np.flatnonzero(m>.5)
        if len(feasible)==0: return 0,0.0
        with torch.no_grad(): z=self.q(torch.as_tensor(np.asarray(state,np.float32)).unsqueeze(0)).squeeze(0).numpy()
        if (not self.eval) and self.rng.random()<self.eps: a=int(self.rng.choice(feasible))
        else: z[m<=.5]=-1e9; a=int(np.argmax(z))
        if m[a]<=.5: self.invalid+=1; a=int(feasible[0])
        return a,0.0
    def get_actions(self,obs,masks): return {n:[self.act_one(s,m)[0] for s,m in zip(obs.get(n,[]),masks.get(n,[]))] for n in range(self.cfg.N_SATS)}
    def record_task_transition(self,sat_id,slot_t,state,action,log_prob,mask,task_reward=0.0):
        if not self.eval: self.tasks.append((int(sat_id),np.asarray(state,np.float32),int(action),np.asarray(mask,np.float32)))
    def _update(self,rewards):
        if not self.tasks: return
        N=self.cfg.N_SATS; D=self.cfg.get_state_dim(); A=self.cfg.get_action_dim()
        states=np.zeros((N,D),np.float32); masks=np.ones((N,A),np.float32); actions=np.zeros(N,np.int64); active=np.zeros(N,np.float32); seen=set()
        for n,s,a,m in self.tasks:
            if n in seen: continue
            seen.add(n); states[n]=s; masks[n]=m; actions[n]=a; active[n]=1
        g=np.mean(np.stack(list(self.cobs.values())),axis=0).astype(np.float32)
        st=torch.as_tensor(states).unsqueeze(0); mt=torch.as_tensor(masks).unsqueeze(0); at=torch.as_tensor(actions).unsqueeze(0); ac=torch.as_tensor(active).unsqueeze(0); gt=torch.as_tensor(g).unsqueeze(0)
        z=self.q(st).masked_fill(mt<=.5,-1e9); chosen=z.gather(-1,at.unsqueeze(-1)).squeeze(-1)*ac; qtot=self.mix(chosen,gt)
        target=torch.tensor([float(np.mean(list(rewards.values())))],dtype=torch.float32)
        loss=F.mse_loss(qtot,target); self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(self.q.parameters())+list(self.mix.parameters()),5.0); self.opt.step(); self.update_count+=1
        if not math.isfinite(float(loss.item())): raise RuntimeError('QMIX non-finite loss')
    def run_step(self,env):
        _,r,d,i=env.step(policy=self)
        if not self.eval: self._update(r)
        return r,d,i

def finite(m): return all(bool(torch.isfinite(p).all()) for p in m.parameters())

def run(name):
    seed_all(SEED); cfg=SmokeConfig(); env=SatelliteMECEnv(cfg); env.reset('train',{'task':3101,'task_param':3102,'dod_init':3103})
    if name=='MAPPO': p=SharedMAPPO(cfg,name='MAPPO'); p.set_train_mode()
    elif name=='IPPO': p=IPPO(cfg); p.set_train_mode()
    elif name=='MADDPG': p=SharedMADDPG(cfg,env,name='MADDPG',seed=SEED,batch_size=16,start_steps=16,updates_per_slot=1,replay_size=3000,reward_scale=1.0,expl_noise=.15); p.set_train_mode()
    else: p=SmokeQMIX(cfg); p.set_train_mode()
    tr=[]; boundaries=0
    for _ in range(TRAIN_SLOTS):
        r,d,i=p.run_step(env); x=float(np.mean(list(r.values()))); assert math.isfinite(x); tr.append(x); boundaries+=int(d)
    if name in ('MAPPO','IPPO'): updates=p.trainer.update_count; ok=finite(p.actor) and finite(p.critic)
    elif name=='MADDPG': updates=p.smoke_updates; ok=finite(p.actor) and finite(p.critic)
    else: updates=p.update_count; ok=finite(p.q) and finite(p.mix) and p.invalid==0
    if updates<=0 or not ok: raise RuntimeError(f'{name}: updates={updates}, finite={ok}')
    ev=SatelliteMECEnv(cfg); ev.reset('eval',{'task':4101,'task_param':4102,'dod_init':4103}); p.set_eval_mode(); er=[]; last={}
    for _ in range(EVAL_SLOTS):
        r,d,last=p.run_step(ev); er.append(float(np.mean(list(r.values()))))
    return {'status':'PASS','updates':int(updates),'rollout_boundaries':boundaries,'mean_train_Rsys':float(np.mean(tr)),'mean_eval_Rsys':float(np.mean(er)),'eval_completion_rate':float(ev.get_eval_completion_rate()),'all_params_finite':True}

def main():
    os.makedirs(OUT,exist_ok=True)
    out={'stage':'real SatelliteMECEnv smoke','seed':SEED,'shared_reward':'R_sys=mean(per-satellite env rewards)','beta_task':0.0,'qmix_note':'slot-level smoke adapter only; micro-step QMIX required before pilot'}
    for n in ('MAPPO','IPPO','MADDPG','QMIX'):
        print('\n===',n,'===',flush=True); out[n]=run(n); print(out[n],flush=True)
    path=os.path.join(OUT,'real_env_smoke_results.json'); json.dump(out,open(path,'w'),indent=2); print('ALL PASS',path)
if __name__=='__main__': main()
