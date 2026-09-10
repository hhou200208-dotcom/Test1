"""Paper-faithful new BLA-MAPPO implementation.

The policy import is lazy so the mathematical utilities remain usable in
environments that have not installed PyTorch yet.
"""

from .config import NewBLAMAPPOConfig

__all__ = ["NewBLAMAPPOConfig", "NewBLAMAPPOPolicy"]


def __getattr__(name):
    if name == "NewBLAMAPPOPolicy":
        from .policy import NewBLAMAPPOPolicy
        return NewBLAMAPPOPolicy
    raise AttributeError(name)

