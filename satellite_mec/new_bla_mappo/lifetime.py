"""Exact secant lifetime-loss equations used by the BLA-MAPPO paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def lifetime_curve(dod: float, a_coef: float = 0.8) -> float:
    """Equation (10): L(D) = D * 10 ** (A * (D - 1))."""
    d = float(np.clip(dod, 0.0, 1.0))
    return d * (10.0 ** (a_coef * (d - 1.0)))


def dod_from_battery(energy: float, capacity: float) -> float:
    """Convert battery energy to DoD without applying a scheduling threshold."""
    if capacity <= 0.0:
        raise ValueError("capacity must be positive")
    return float(np.clip(1.0 - energy / capacity, 0.0, 1.0))


def net_discharge_loss(dod_start: float, dod_end: float, a_coef: float = 0.8) -> float:
    """Psi(Da, Db) = [L(Db) - L(Da)]+."""
    return max(lifetime_curve(dod_end, a_coef) - lifetime_curve(dod_start, a_coef), 0.0)


def induced_lifetime_loss(
    dod_start: float,
    battery_actual: float,
    battery_base: float,
    capacity: float,
    a_coef: float = 0.8,
) -> float:
    """Equation (12), separating task load from platform base operation."""
    d_actual = dod_from_battery(battery_actual, capacity)
    d_base = dod_from_battery(battery_base, capacity)
    return max(
        net_discharge_loss(dod_start, d_actual, a_coef)
        - net_discharge_loss(dod_start, d_base, a_coef),
        0.0,
    )


@dataclass
class EmpiricalMeanNormalizer:
    """Mean scale estimated during the first ``warmup_slots`` and then frozen."""

    warmup_slots: int = 100
    eps: float = 1e-8
    total: float = 0.0
    count: int = 0
    slots: int = 0
    frozen_value: float | None = None

    @property
    def value(self) -> float:
        if self.frozen_value is not None:
            return self.frozen_value
        return max(self.total / max(self.count, 1), self.eps)

    @property
    def frozen(self) -> bool:
        return self.frozen_value is not None

    def observe_slot(self, values: Iterable[float]) -> None:
        if self.frozen:
            return
        vals = [max(float(v), 0.0) for v in values]
        self.total += sum(vals)
        self.count += len(vals)
        self.slots += 1
        if self.slots >= self.warmup_slots:
            self.freeze()

    def freeze(self) -> None:
        if self.frozen_value is None:
            self.frozen_value = max(self.total / max(self.count, 1), self.eps)

    def state_dict(self) -> dict:
        return {
            "warmup_slots": self.warmup_slots,
            "eps": self.eps,
            "total": self.total,
            "count": self.count,
            "slots": self.slots,
            "frozen_value": self.frozen_value,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("total", "count", "slots", "frozen_value"):
            if key in state:
                setattr(self, key, state[key])
