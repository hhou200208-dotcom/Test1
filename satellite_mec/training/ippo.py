"""Independent PPO with the exact MAPPO actor and an explicit local critic."""
import torch
from training.networks import Actor, Critic, RolloutBuffer
from training.policy import MAPPOPolicy
from training.trainer import MAPPOTrainer

class _LocalConfig:
    def __init__(self, cfg): self._cfg = cfg
    def __getattr__(self, key): return getattr(self._cfg, key)
    def get_critic_state_dim(self): return self._cfg.get_state_dim() - 7

class IPPOTrainer(MAPPOTrainer):
    def __init__(self, config):
        super().__init__(_LocalConfig(config))
        self.cfg = config
    def compute_bootstrap(self, env, buffer):
        out = {}
        for sid, obs in env.get_local_critic_obs().items():
            with torch.no_grad(): out[sid] = float(self.critic.get_value(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).item())
        return out

class IPPOPolicy(MAPPOPolicy):
    def __init__(self, config, name="IPPO"):
        self.cfg=config; self.name=name; self.lyapunov_calc=None
        self.trainer=IPPOTrainer(config); self.buffer=RolloutBuffer(config)
        self.actor=self.trainer.actor; self.critic=self.trainer.critic
        self.learning_curve=[]; self._slot_inference={}; self._slot_critic={}; self._current_slot_t=0; self._eval_mode=False
    def collect_critic_values(self, env):
        self._slot_critic = {}
        for sid, obs in env.get_local_critic_obs().items():
            with torch.no_grad(): value=self.critic.get_value(torch.as_tensor(obs, dtype=torch.float32, device=self.trainer.device))
            self._slot_critic[sid]=(obs, float(value.item()))
