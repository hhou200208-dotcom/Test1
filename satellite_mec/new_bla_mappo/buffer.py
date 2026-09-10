"""Trajectory buffer implementing equations (31)-(36) of the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PaperSlotRecord:
    sat_id: int
    slot_t: int
    critic_state: np.ndarray
    system_reward: float
    value: float
    advantage: float = 0.0
    target_return: float = 0.0


@dataclass
class PaperTaskRecord:
    sat_id: int
    slot_t: int
    state: np.ndarray
    action: int
    log_prob: float
    mask: np.ndarray
    structured_cost: float
    advantage: float = 0.0
    centered_cost: float = 0.0


class PaperRolloutBuffer:
    """Continuous-state rollout with system GAE and bounded credit refinement."""

    def __init__(self, config):
        self.cfg = config
        self.slots: List[PaperSlotRecord] = []
        self.tasks: List[PaperTaskRecord] = []
        self._slot_index: Dict[Tuple[int, int], PaperSlotRecord] = {}

    def add_slot(self, sat_id, slot_t, critic_state, system_reward, value) -> None:
        record = PaperSlotRecord(
            sat_id=int(sat_id), slot_t=int(slot_t),
            critic_state=np.asarray(critic_state, np.float32).copy(),
            system_reward=float(system_reward), value=float(value),
        )
        self.slots.append(record)
        self._slot_index[(record.sat_id, record.slot_t)] = record

    def add_task(self, sat_id, slot_t, state, action, log_prob, mask,
                 structured_cost) -> None:
        self.tasks.append(PaperTaskRecord(
            sat_id=int(sat_id), slot_t=int(slot_t),
            state=np.asarray(state, np.float32).copy(), action=int(action),
            log_prob=float(log_prob), mask=np.asarray(mask, np.float32).copy(),
            structured_cost=float(structured_cost),
        ))

    def compute_advantages(self, bootstrap_values: Dict[int, float]) -> None:
        """System-level GAE followed by equations (35)-(36)."""
        by_sat: Dict[int, List[PaperSlotRecord]] = {}
        for record in self.slots:
            by_sat.setdefault(record.sat_id, []).append(record)

        gamma = self.cfg.GAMMA
        lam = self.cfg.LAMBDA_GAE
        for sat_id, records in by_sat.items():
            records.sort(key=lambda r: r.slot_t)
            next_value = float(bootstrap_values.get(sat_id, 0.0))
            gae = 0.0
            # Rollout boundaries are not terminals in the continuing system.
            for record in reversed(records):
                delta = record.system_reward + gamma * next_value - record.value
                gae = delta + gamma * lam * gae
                record.advantage = gae
                record.target_return = gae + record.value
                next_value = record.value

        mapped = np.asarray([
            self._slot_index[(t.sat_id, t.slot_t)].advantage for t in self.tasks
        ], dtype=np.float32)
        if mapped.size == 0:
            return
        mean_adv = float(mapped.mean())
        std_adv = float(mapped.std())

        task_groups: Dict[Tuple[int, int], List[PaperTaskRecord]] = {}
        for task in self.tasks:
            task_groups.setdefault((task.sat_id, task.slot_t), []).append(task)

        eta_c = self.cfg.BLA_ETA_C
        for key, records in task_groups.items():
            mean_cost = float(np.mean([r.structured_cost for r in records]))
            sys_adv = self._slot_index[key].advantage
            normalized_sys = (sys_adv - mean_adv) / (std_adv + 1e-8)
            for record in records:
                # Eq. (28): factor 1/2 preserves the original magnitude bound.
                record.centered_cost = 0.5 * (record.structured_cost - mean_cost)
                record.advantage = normalized_sys - eta_c * record.centered_cost

    def actor_batch(self) -> dict:
        if not self.tasks:
            return {}
        return {
            "states": np.asarray([r.state for r in self.tasks], np.float32),
            "actions": np.asarray([r.action for r in self.tasks], np.int64),
            "log_probs_old": np.asarray([r.log_prob for r in self.tasks], np.float32),
            "masks": np.asarray([r.mask for r in self.tasks], np.float32),
            "advantages": np.asarray([r.advantage for r in self.tasks], np.float32),
        }

    def critic_batch(self) -> dict:
        if not self.slots:
            return {}
        return {
            "states": np.asarray([r.critic_state for r in self.slots], np.float32),
            "returns": np.asarray([r.target_return for r in self.slots], np.float32),
        }

    def clear(self) -> None:
        self.slots.clear()
        self.tasks.clear()
        self._slot_index.clear()

    def __len__(self) -> int:
        return len(self.tasks)

