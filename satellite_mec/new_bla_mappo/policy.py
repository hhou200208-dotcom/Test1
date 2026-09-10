"""New BLA-MAPPO policy following Algorithms 1-2 in ``test6.pdf``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from core.dvfs import select_freq
from interfaces import PolicyInterface

from .buffer import PaperRolloutBuffer
from .lifetime import EmpiricalMeanNormalizer, induced_lifetime_loss
from .trainer import NewBLAMAPPOTrainer


@dataclass
class _SlotTracker:
    energy_raw: float
    q_cycles_next: float
    nb_next: int
    floor_next: float
    link_remaining: Dict[int, float]


class NewBLAMAPPOPolicy(PolicyInterface):
    """Battery Lifetime-Aware MAPPO with dual-granularity credit assignment.

    This class deliberately does not reuse the legacy ``MAPPOPolicy`` reward or
    buffer. It uses the paper's system reward, exact secant lifetime loss, and
    structured task cost in equations (22)-(28) and (31)-(38).
    """

    name = "new_BLA-MAPPO"
    needs_training = True

    def __init__(self, config):
        self.cfg = config
        self.trainer = NewBLAMAPPOTrainer(config)
        self.actor = self.trainer.actor
        self.critic = self.trainer.critic
        self.buffer = PaperRolloutBuffer(config)
        self._eval_mode = False
        self._normalization_only = False
        self._trackers: Dict[int, _SlotTracker] = {}
        self._slot_critic: Dict[int, tuple[np.ndarray, float]] = {}
        self._slot_task_lifetimes: List[float] = []
        self._last_lifetime_mean = 0.0
        self.learning_curve: List[dict] = []
        self.slot_lifetime_norm = EmpiricalMeanNormalizer(
            config.BLA_T_NORM, config.BLA_NORM_EPS)
        self.task_lifetime_norm = EmpiricalMeanNormalizer(
            config.BLA_T_NORM, config.BLA_NORM_EPS)

    # ------------------------------------------------------------------
    # Algorithm 1: slot resource tracking, observation and masking
    # ------------------------------------------------------------------
    def admit_task(self, env, sat, task) -> bool:
        """Apply Table 2's per-satellite admission threshold theta_n."""
        if len(sat.forward_queue) >= self.cfg.BLA_ADMISSION_TASKS:
            return False
        sat.forward_queue.append(task)
        sat.qf_size += task.size
        return True

    def begin_slot(self, env) -> None:
        t = env.current_slot
        cfg = self.cfg
        self._trackers = {}
        self._slot_task_lifetimes = []
        for sat in env.constellation.satellites:
            nb = len(sat.compute_queue)
            q_cycles = sum(task.get_remaining_size() * task.cpu_cycles
                           for task in sat.compute_queue)
            floor = self._deadline_floor_from_cycles(sat.compute_queue, t, nb)
            f_cmp = select_freq(cfg, q_cycles, f_floor=floor) if nb else 0.0
            comp_energy = cfg.KAPPA * (f_cmp ** 3) * cfg.TAU

            # Predict the retained next-slot compute state after this slot's
            # equal-share service, matching Satellite.process_tasks().
            served_each = f_cmp * cfg.TAU / max(nb, 1)
            retained_cycles: List[tuple[float, float]] = []
            for task in sat.compute_queue:
                remaining = task.get_remaining_size() * task.cpu_cycles
                after = max(remaining - served_each, 0.0)
                if after > 0.0:
                    retained_cycles.append((after, max(task.remain_time(t + 1), cfg.TAU)))
            q_next = sum(item[0] for item in retained_cycles)
            nb_next = len(retained_cycles)
            floor_next = (nb_next * max((c / w for c, w in retained_cycles), default=0.0))

            solar_now = float(sat.solar_seq[min(t, len(sat.solar_seq) - 1)]) * cfg.TAU
            battery = cfg.E_CAP * (1.0 - sat.dod)
            gamma = battery + solar_now - cfg.P_HOUSEKEEPING * cfg.TAU - comp_energy
            links = {
                nid: max(sat.link_rates[nid] * (cfg.TAU - sat.prop_delays[nid]), 0.0)
                for nid in sat.neighbors
            }
            self._trackers[sat.sat_id] = _SlotTracker(
                energy_raw=gamma,
                q_cycles_next=q_next,
                nb_next=nb_next,
                floor_next=floor_next,
                link_remaining=links,
            )

    def _deadline_floor_from_cycles(self, tasks, current_slot: int, n_tasks: int) -> float:
        if not tasks or n_tasks <= 0:
            return 0.0
        need = max(
            task.get_remaining_size() * task.cpu_cycles
            / max(task.remain_time(current_slot), self.cfg.TAU)
            for task in tasks
        )
        return n_tasks * need

    def _next_solar(self, sat, t: int) -> float:
        idx = min(t + 1, len(sat.solar_seq) - 1)
        return float(sat.solar_seq[idx]) * self.cfg.TAU

    def _predicted_comp_energy(self, tracker: _SlotTracker) -> tuple[float, int, float, float]:
        q_cycles = tracker.q_cycles_next
        n_tasks = tracker.nb_next
        floor = tracker.floor_next
        f_cmp = select_freq(self.cfg, q_cycles, f_floor=floor) if n_tasks else 0.0
        energy = self.cfg.KAPPA * (f_cmp ** 3) * self.cfg.TAU
        return energy, n_tasks, q_cycles, floor

    def _local_prediction(self, sat, task, t: int) -> tuple[float, int, float, float]:
        tracker = self._trackers[sat.sat_id]
        task_cycles = task.get_remaining_size() * task.cpu_cycles
        n_after = tracker.nb_next + 1
        q_after = tracker.q_cycles_next + task_cycles
        floor_after = max(
            tracker.floor_next,
            n_after * task_cycles / max(task.remain_time(t + 1), self.cfg.TAU),
        )
        f_after = select_freq(self.cfg, q_after, f_floor=floor_after)
        e_after = self.cfg.KAPPA * (f_after ** 3) * self.cfg.TAU
        return e_after, n_after, q_after, floor_after

    def build_action_mask(self, env, sat, task, t, neighbor_info, neighbor_nb) -> np.ndarray:
        cfg = self.cfg
        tracker = self._trackers[sat.sat_id]
        mask = np.zeros(cfg.get_action_dim(), dtype=np.float32)
        if tracker.energy_raw < cfg.BLA_B_MIN:
            return mask

        # Eq. (19): next-slot compute capacity and energy feasibility.
        e_after, n_after, _, _ = self._local_prediction(sat, task, t)
        local_margin = (
            min(cfg.E_CAP, tracker.energy_raw)
            + self._next_solar(sat, t)
            - cfg.P_HOUSEKEEPING * cfg.TAU
            - e_after
            - cfg.BLA_B_MIN
        )
        if n_after <= cfg.MAX_DISPATCH and local_margin >= 0.0:
            mask[0] = 1.0

        # Eq. (20)-(21): link, hops, delay, current and next-slot energy.
        next_comp, _, _, _ = self._predicted_comp_energy(tracker)
        for idx, neighbor_id in enumerate(sat.neighbors):
            rate = sat.link_rates[neighbor_id]
            tx_energy = cfg.P_T * task.size / max(rate, 1e-12)
            energy_after_tx = tracker.energy_raw - tx_energy
            next_margin = (
                min(cfg.E_CAP, energy_after_tx)
                + self._next_solar(sat, t)
                - cfg.P_HOUSEKEEPING * cfg.TAU
                - next_comp
                - cfg.BLA_B_MIN
            )
            feasible_delay = task.feasible_forward(
                rate, sat.prop_delays[neighbor_id],
                neighbor_nb.get(neighbor_id, 0), cfg.CPU_FREQ,
                cfg.TAU, t, cfg.K_MAX,
            )
            if (
                task.size <= tracker.link_remaining[neighbor_id]
                and task.hops < cfg.K_MAX
                and feasible_delay
                and energy_after_tx >= cfg.BLA_B_MIN
                and next_margin >= 0.0
            ):
                mask[idx + 1] = 1.0
        return mask

    def _base_observation(self, env, sat, t: int) -> np.ndarray:
        cfg = self.cfg
        tracker = self._trackers[sat.sat_id]
        local = [
            sat.qf_size / max(cfg.Q_F_MAX, 1.0),
            tracker.nb_next / max(cfg.MAX_DISPATCH, 1),
            sat.dod / max(cfg.DOD_MAX, 1e-8),
            float(np.clip((tracker.energy_raw - cfg.BLA_B_MIN) / cfg.E_CAP, -1.0, 1.0)),
            float(float(sat.solar_seq[min(t, len(sat.solar_seq) - 1)]) > 0.0),
            sat.tau_switch / max(cfg.ORBIT_PERIOD, 1),
        ]
        neighbors: List[float] = []
        max_prop = cfg.ORBIT_RADIUS / cfg.SPEED_OF_LIGHT
        for nid in sat.neighbors:
            other = env.constellation.satellites[nid]
            other_tracker = self._trackers[nid]
            neighbors.extend([
                other.qf_size / max(cfg.Q_F_MAX, 1.0),
                other_tracker.nb_next / max(cfg.MAX_DISPATCH, 1),
                other.dod / max(cfg.DOD_MAX, 1e-8),
                max(other_tracker.energy_raw - cfg.BLA_B_MIN, 0.0) / cfg.E_CAP,
                sat.link_rates[nid] / cfg.B_MAX,
                sat.prop_delays[nid] / max(max_prop, 1e-12),
                tracker.link_remaining[nid] / max(cfg.B_MAX * cfg.TAU, 1.0),
            ])
        return np.asarray(local + neighbors, dtype=np.float32)

    def build_actor_state(self, env, sat, task, t, neighbor_info) -> np.ndarray:
        cfg = self.cfg
        remain = max(task.remain_time(t), 0.0)
        freq_need = task.size * task.cpu_cycles / max(remain, cfg.TAU) / cfg.CPU_FREQ
        task_features = np.asarray([
            task.size / cfg.S_MAX,
            task.cpu_cycles / cfg.H_MAX,
            task.hops / max(cfg.K_MAX, 1),
            remain / cfg.D_MAX_MAX,
            min(remain / cfg.D_MAX_MAX, 1.0),
            min(freq_need, 1.0),
        ], dtype=np.float32)
        state = np.concatenate([self._base_observation(env, sat, t), task_features])
        if state.shape != (cfg.get_state_dim(),):
            raise RuntimeError(f"actor state shape {state.shape}, expected {cfg.get_state_dim()}")
        return state

    # ------------------------------------------------------------------
    # Equations (25)-(28): task-level structured cost
    # ------------------------------------------------------------------
    def task_structured_cost(self, env, sat, task, action, t, neighbor_info,
                             commit: bool = False) -> float:
        cfg = self.cfg
        tracker = self._trackers[sat.sat_id]
        source_load = len(sat.forward_queue) / max(cfg.BLA_ADMISSION_TASKS, 1)
        battery_start_next = min(cfg.E_CAP, tracker.energy_raw)

        if action == 0:
            e_before, _, _, _ = self._predicted_comp_energy(tracker)
            e_after, n_after, q_after, floor_after = self._local_prediction(sat, task, t)
            common = battery_start_next + self._next_solar(sat, t) - cfg.P_HOUSEKEEPING * cfg.TAU
            delta_l = induced_lifetime_loss(
                1.0 - battery_start_next / cfg.E_CAP,
                common - e_after,
                common - e_before,
                cfg.E_CAP,
                cfg.A_COEF,
            )
            destination_load = n_after / max(cfg.MAX_DISPATCH, 1)
            if commit:
                tracker.nb_next = n_after
                tracker.q_cycles_next = q_after
                tracker.floor_next = floor_after
        else:
            nid = sat.neighbors[action - 1]
            tx_energy = cfg.P_T * task.size / max(sat.link_rates[nid], 1e-12)
            no_action = min(cfg.E_CAP, tracker.energy_raw)
            with_action = min(cfg.E_CAP, tracker.energy_raw - tx_energy)
            delta_l = induced_lifetime_loss(
                sat.dod, with_action, no_action, cfg.E_CAP, cfg.A_COEF)
            destination_load = (
                len(env.constellation.satellites[nid].forward_queue) + 1
            ) / max(cfg.BLA_ADMISSION_TASKS, 1)
            if commit:
                tracker.energy_raw -= tx_energy
                tracker.link_remaining[nid] -= task.size

        queue_cost = destination_load - source_load
        scale = self.task_lifetime_norm.value
        structured = (
            cfg.BLA_LAMBDA_Q * math.tanh(queue_cost)
            + cfg.BLA_LAMBDA_L * math.tanh(delta_l / scale)
        )
        self._slot_task_lifetimes.append(delta_l)
        return float(structured)

    # ------------------------------------------------------------------
    # Equation (12), (22)-(23): true lifetime loss and system reward
    # ------------------------------------------------------------------
    def compute_slot_lifetime_losses(self, env, dod_before_map) -> List[float]:
        cfg = self.cfg
        losses = []
        for sat in env.constellation.satellites:
            dod_before = float(dod_before_map[sat.sat_id])
            battery_before = cfg.E_CAP * (1.0 - dod_before)
            base = cfg.P_HOUSEKEEPING * cfg.TAU
            solar = sat.solar_power * cfg.TAU
            task_energy = sat.slot_comp_energy + sat.slot_trans_energy
            battery_base = min(cfg.E_CAP, battery_before + solar - base)
            battery_actual = min(cfg.E_CAP, battery_before + solar - base - task_energy)
            losses.append(induced_lifetime_loss(
                dod_before, battery_actual, battery_base, cfg.E_CAP, cfg.A_COEF))

        self._last_lifetime_mean = float(np.mean(losses)) if losses else 0.0
        if not self._eval_mode:
            self.slot_lifetime_norm.observe_slot(losses)
            self.task_lifetime_norm.observe_slot(self._slot_task_lifetimes)
        return losses

    def _system_reward(self, info: dict) -> float:
        cfg = self.cfg
        losses = info.get("paper_lifetime_losses") or [0.0] * cfg.N_SATS
        local_rewards = []
        for n in range(cfg.N_SATS):
            local_rewards.append(
                cfg.W_DONE * info["per_sat_satisfied"][n]
                - cfg.W_TIMEOUT * info["per_sat_timeout"][n]
                - cfg.W_REJECT * info["per_sat_rejected"][n]
                - cfg.BLA_RHO_L * losses[n] / self.slot_lifetime_norm.value
            )
        return float(np.mean(local_rewards))

    # ------------------------------------------------------------------
    # Algorithm 2: CTDE collection and PPO update
    # ------------------------------------------------------------------
    def _global_context(self, env, t: int) -> np.ndarray:
        cfg = self.cfg
        sats = env.constellation.satellites
        dod = np.asarray([s.dod for s in sats], dtype=np.float64)
        return np.asarray([
            np.mean(dod) / cfg.DOD_MAX,
            np.std(dod) / cfg.DOD_MAX,
            np.max(dod) / cfg.DOD_MAX,
            np.mean([len(s.forward_queue) for s in sats]) / max(cfg.BLA_ADMISSION_TASKS, 1),
            np.mean([self._trackers[s.sat_id].nb_next for s in sats]) / cfg.MAX_DISPATCH,
            self._last_lifetime_mean / self.slot_lifetime_norm.value,
            np.mean([s.last_cpu_freq for s in sats]) / cfg.CPU_FREQ,
            np.mean([float(s.solar_seq[min(t, len(s.solar_seq) - 1)]) for s in sats]) / cfg.P_SOLAR_MAX,
            np.mean([float(float(s.solar_seq[min(t, len(s.solar_seq) - 1)]) > 0.0) for s in sats]),
            (t % cfg.ORBIT_PERIOD) / cfg.ORBIT_PERIOD,
        ], dtype=np.float32)

    def build_critic_state(self, env, sat, t: int) -> np.ndarray:
        state = np.concatenate([
            self._base_observation(env, sat, t), self._global_context(env, t)
        ])
        if state.shape != (self.cfg.get_critic_state_dim(),):
            raise RuntimeError(
                f"critic state shape {state.shape}, expected {self.cfg.get_critic_state_dim()}")
        return state

    def collect_critic_values(self, env) -> None:
        self._slot_critic = {}
        if self._eval_mode or self._normalization_only:
            return
        for sat in env.constellation.satellites:
            state = self.build_critic_state(env, sat, env.current_slot)
            self._slot_critic[sat.sat_id] = (state, self.trainer.value(state))

    def act_one(self, state: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.trainer.device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.trainer.device)
        action, log_prob, _ = self.actor.get_action(
            state_t, mask_t, deterministic=self._eval_mode)
        return action, log_prob

    def record_paper_task_transition(self, sat_id, slot_t, state, action,
                                     log_prob, mask, structured_cost) -> None:
        if not self._eval_mode and not self._normalization_only:
            self.buffer.add_task(
                sat_id, slot_t, state, action, log_prob, mask, structured_cost)

    def run_step(self, env):
        _, _, done, info = env.step(policy=self)
        system_reward = self._system_reward(info)
        info["new_bla_system_reward"] = system_reward
        info["new_bla_lifetime_norm"] = self.slot_lifetime_norm.value
        info["new_bla_task_lifetime_norm"] = self.task_lifetime_norm.value

        if not self._eval_mode:
            for sat_id, (state, value) in self._slot_critic.items():
                self.buffer.add_slot(sat_id, info["slot"], state, system_reward, value)
            if done and len(self.buffer) > 0:
                self._update(env)
        return {n: system_reward for n in range(self.cfg.N_SATS)}, done, info

    def _bootstrap_values(self, env) -> Dict[int, float]:
        self.begin_slot(env)
        return {
            sat.sat_id: self.trainer.value(
                self.build_critic_state(env, sat, env.current_slot))
            for sat in env.constellation.satellites
        }

    def _update(self, env) -> dict:
        metrics = self.trainer.update(self.buffer, self._bootstrap_values(env))
        self.buffer.clear()
        return metrics

    def get_actions(self, obs, masks):
        actions = {}
        for sat_id, states in obs.items():
            actions[sat_id] = [self.act_one(s, m)[0] for s, m in zip(states, masks[sat_id])]
        return actions

    def set_eval_mode(self) -> None:
        self._eval_mode = True
        self.actor.eval()
        self.critic.eval()

    def set_train_mode(self) -> None:
        self._eval_mode = False
        self.actor.train()
        self.critic.train()

    def quick_eval(self, env, n_slots: int = 1000):
        from core.env import SatelliteMECEnv

        was_eval = self._eval_mode
        eval_env = SatelliteMECEnv(self.cfg)
        eval_env.reset(phase="eval", seeds=self.cfg.get_quick_eval_seeds())
        self.set_eval_mode()
        dods, losses = [], []
        for _ in range(n_slots):
            _, _, info = self.run_step(eval_env)
            dods.append(info["avg_dod"])
            losses.append(np.mean(info.get("paper_lifetime_losses") or [0.0]))
        result = (
            eval_env.get_eval_completion_rate(),
            float(np.mean(dods)),
            float(np.mean(losses)),
        )
        self.learning_curve.append({
            "update": self.trainer.update_count,
            "completion_rate": result[0], "avg_dod": result[1],
            "avg_lifetime_loss": result[2],
        })
        if not was_eval:
            self.set_train_mode()
        return result

    def calibrate_normalizers(self, env, n_slots: int | None = None) -> None:
        """Collect the paper's first-T_norm empirical scales, then freeze them."""
        if self.slot_lifetime_norm.frozen and self.task_lifetime_norm.frozen:
            return
        slots = int(n_slots or self.cfg.BLA_T_NORM)
        self._normalization_only = True
        self.set_train_mode()
        env.reset(phase="warmup", seeds={
            "task": self.cfg.SEED + 700,
            "task_param": self.cfg.SEED + 701,
            "dod_init": self.cfg.SEED + 702,
        })
        for _ in range(slots):
            env.step(policy=self)
        self.slot_lifetime_norm.freeze()
        self.task_lifetime_norm.freeze()
        self.buffer.clear()
        self._normalization_only = False

    def save(self, path: str, extra_info: dict | None = None) -> None:
        extra = {
            "learning_curve": self.learning_curve,
            "slot_lifetime_norm": self.slot_lifetime_norm.state_dict(),
            "task_lifetime_norm": self.task_lifetime_norm.state_dict(),
            **(extra_info or {}),
        }
        self.trainer.save(path, extra)

    def load(self, path: str) -> dict:
        progress = self.trainer.load(path)
        self.learning_curve = progress.get("learning_curve", [])
        self.slot_lifetime_norm.load_state_dict(progress.get("slot_lifetime_norm", {}))
        self.task_lifetime_norm.load_state_dict(progress.get("task_lifetime_norm", {}))
        return progress
