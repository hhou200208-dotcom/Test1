import copy
import numpy as np
import torch
from core.common_reward import COMPONENTS, CommonRewardConfig, common_reward_v1, config_from_env
from core.config import Config
from core.env import SatelliteMECEnv
from training.policy import MAPPOPolicy
from training.ippo import IPPOPolicy
from baselines.maddpg_dod import MADDPGDoDPolicy
from baselines.td3_sched import TD3SchedPolicy

def test_common_reward_config_and_exact_total():
    c=Config(); rc=config_from_env(c)
    assert rc == CommonRewardConfig(10,5,5,2,.05)
    x=common_reward_v1(-1.25,2,1,3,.4,.7,rc)
    assert x["total"] == sum(x[k] for k in COMPONENTS)
    assert x["action_cost"] == -1.25 and x["done"] == 20

def test_train_eval_reward_definition_is_identical():
    ledgers=[]
    for phase in ("train","eval"):
        c=Config(); e=SatelliteMECEnv(c); e.reset(phase,{"task":77,"task_param":78,"dod_init":79})
        _,r,_,info=e.step(actions={})
        assert np.isclose(sum(r.values()),info["reward_ledger"]["total"])
        ledgers.append(info["reward_ledger"])
    assert ledgers[0] == ledgers[1]

def test_explicit_ippo_observation_and_actor_fairness():
    c=Config(); e=SatelliteMECEnv(c); e.reset()
    assert all(x.shape==(47,) for x in e.get_local_critic_obs().values())
    torch.manual_seed(12); m=MAPPOPolicy(c)
    torch.manual_seed(12); i=IPPOPolicy(c)
    assert type(m.actor) is type(i.actor)
    assert sum(p.numel() for p in m.actor.parameters()) == sum(p.numel() for p in i.actor.parameters())
    assert m.critic.input_norm.normalized_shape==(245,) and i.critic.input_norm.normalized_shape==(47,)

def test_continuous_policies_masked_deterministic_actions():
    c=Config(); e=SatelliteMECEnv(c); s=np.zeros(54,np.float32); mask=np.array([1,0,1,0,0],np.float32)
    for p in (MADDPGDoDPolicy(c,e,seed=1),TD3SchedPolicy(c,e,seed=1)):
        p.set_eval_mode(); a,_=p.act_one(s,mask); assert mask[a] == 1
