"""
core/dvfs.py
============
Constellation-wide DVFS (Dynamic Voltage and Frequency Scaling) controller.

Adapted from Li et al. IEEE TSC 2024, Eq.(8): E_cmp(t) = a · f_cmp(t)^3 · T_d.
Every slot, each satellite selects a single f_cmp(t) shared by all locally
processed tasks, via a Lyapunov drift-plus-penalty closed-form rule.

Derivation
----------
Per-slot drift-plus-penalty objective:
    min_f  V · κ · f^3 · τ  −  Q_cmp · f · τ
        s.t.  0 ≤ f ≤ F_CMP_MAX

where Q_cmp is the remaining computation backlog (in cycles).
First-order condition:
    3 V κ f^2 τ − Q_cmp τ = 0  ⇒  f* = sqrt( Q_cmp / (3 V κ) )

V is auto-calibrated in Config so that f* hits F_CMP_MAX exactly at the
"saturated" backlog Q_max = MAX_DISPATCH · S_MAX · H_MAX (worst-case queue).
"""

import math


def select_freq(cfg, q_cycles_remaining: float,
                f_floor: float = 0.0) -> float:
    """Closed-form Lyapunov DVFS with a deadline floor.

    f_cmp = clip( max(f_lyapunov, f_floor),  0, F_CMP_MAX )

    where
        f_lyapunov = sqrt( Q_cmp / (3 V κ) )
        f_floor    = nb · max over tasks (remaining_cycles_i / remaining_time_i)

    The Lyapunov term saves energy when load is light; the floor guarantees
    that every queued task can finish before its deadline at the current
    nb-task equal-share schedule (otherwise DVFS would starve the queue).

    Parameters
    ----------
    cfg                 : Config (uses V_DVFS, KAPPA, CPU_FREQ).
    q_cycles_remaining  : Sum of remaining CPU cycles across all tasks
                          in the satellite's compute_queue.
    f_floor             : Deadline-driven minimum frequency (cycles/s).

    Returns
    -------
    f_cmp : float, cycles/s, clipped to [0, CPU_FREQ].
    """
    if q_cycles_remaining <= 0.0:
        return 0.0
    f_lyap = math.sqrt(q_cycles_remaining / (3.0 * cfg.V_DVFS * cfg.KAPPA))
    return min(max(f_lyap, f_floor), cfg.CPU_FREQ)


def deadline_floor(cfg, compute_queue, current_slot: int) -> float:
    """Compute the deadline-driven f_cmp floor.

    Under equal-share scheduling, each of nb tasks gets f_cmp/nb cycles/s.
    For task i to finish: remaining_cycles_i / (f_cmp / nb) ≤ remaining_time_i,
    i.e. f_cmp ≥ nb · remaining_cycles_i / remaining_time_i.
    Take the worst task as the floor.
    """
    nb = len(compute_queue)
    if nb == 0:
        return 0.0
    floor = 0.0
    for task in compute_queue:
        remain_cycles = task.get_remaining_size() * task.cpu_cycles
        remain_time   = max(task.remain_time(current_slot), cfg.TAU)
        floor = max(floor, nb * remain_cycles / remain_time)
    return floor
