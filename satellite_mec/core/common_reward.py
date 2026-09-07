"""Canonical reward used by the fair RL comparison experiment."""
from dataclasses import asdict, dataclass

REWARD_DEFINITION = "common_reward_v1"
COMPONENTS = ("action_cost", "done", "timeout", "reject", "health_loss", "queue")

@dataclass(frozen=True)
class CommonRewardConfig:
    w_done: float = 10.0
    w_timeout: float = 5.0
    w_reject: float = 5.0
    w_health_loss: float = 2.0
    w_queue: float = 0.05
    def to_dict(self): return asdict(self)

def common_reward_v1(action_cost, done, timeout, reject, health_loss,
                     queue_pressure, cfg):
    ledger = {"action_cost": float(action_cost), "done": cfg.w_done * int(done),
              "timeout": -cfg.w_timeout * int(timeout),
              "reject": -cfg.w_reject * int(reject),
              "health_loss": -cfg.w_health_loss * float(health_loss),
              "queue": -cfg.w_queue * float(queue_pressure)}
    ledger["total"] = float(sum(ledger[k] for k in COMPONENTS))
    return ledger

def config_from_env(cfg):
    return CommonRewardConfig(cfg.W_DONE, cfg.W_TIMEOUT, cfg.W_REJECT, cfg.W_HL, cfg.W_QUEUE)
