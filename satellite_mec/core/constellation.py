"""
core/constellation.py
======================
25颗卫星组成的Walker星座，管理拓扑、链路、太阳能和任务生成。

使用示例
--------
    from core.config import Config
    from core.constellation import Constellation

    cfg   = Config()
    const = Constellation(cfg)
    const.reset(seeds={'task': 42, 'task_param': 43, 'dod_init': 44})

    tasks_by_sat = const.generate_tasks(t=0)
    global_info  = const.exchange_info()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from core.satellite import Satellite
from core.task import Task

if TYPE_CHECKING:
    from core.config import Config


class Constellation:
    """
    Walker星座（5平面×5颗卫星 = 25颗）。

    职责
    ----
    - 构建4近邻网格拓扑
    - 初始化星间链路速率和传播时延
    - 预计算每颗卫星的太阳能时序
    - 管理任务生成（泊松到达）和多跳投递
    """

    def __init__(self, config: "Config"):
        self.cfg = config
        self.rng_task       = np.random.default_rng(config.SEED_TASK)
        self.rng_task_param = np.random.default_rng(config.SEED_TASK_PARAM)
        self.rng_link       = np.random.default_rng(config.SEED_LINK)
        self.rng_dod        = np.random.default_rng(config.SEED)

        self.neighbor_matrix  = self._build_topology()
        self.link_rate_matrix = self._init_link_rates()
        self.prop_delay_matrix = self._init_prop_delays()
        self.solar_seq        = self._precompute_solar(config.T_TOTAL)

        # 高/低负载卫星划分
        rng_load  = np.random.default_rng(config.SEED_LINK + 10)
        n_high    = int(config.N_SATS * config.LAMBDA_HIGH_RATIO)
        high_sats = rng_load.choice(config.N_SATS, size=n_high, replace=False)
        self.high_load_sats = set(high_sats.tolist())
        self.low_load_sats  = set(range(config.N_SATS)) - self.high_load_sats

        self.satellites: List[Satellite] = self._create_satellites()
        self._task_counter: Dict[Tuple, int] = {}

        print(f"[Constellation] 高负载卫星({n_high}颗): {sorted(self.high_load_sats)}")
        print(f"[Constellation] 低负载卫星({config.N_SATS - n_high}颗): "
              f"{sorted(self.low_load_sats)}")

    # ── 拓扑构建 ──────────────────────────────────────────────
    def _build_topology(self) -> np.ndarray:
        """
        构建4近邻网格：[前、后、左轨、右轨]。

        Returns
        -------
        np.ndarray : shape=(N_SATS, 4)，每行是4个邻居的sat_id
        """
        cfg = self.cfg
        P, S, N = cfg.N_PLANES, cfg.N_SATS_PER_PLANE, cfg.N_SATS
        m = np.zeros((N, 4), dtype=np.int32)
        for n in range(N):
            p, s = n // S, n % S
            m[n] = [p*S+(s+1) % S, p*S+(s-1) % S,
                    ((p-1) % P)*S+s, ((p+1) % P)*S+s]
        return m

    def _init_link_rates(self) -> np.ndarray:
        """初始化星间链路速率（双向对称）。"""
        cfg = self.cfg
        N   = cfg.N_SATS
        lrm = np.zeros((N, 4), dtype=np.float64)
        processed = set()
        for n in range(N):
            for idx in range(4):
                m    = self.neighbor_matrix[n, idx]
                pair = tuple(sorted([n, m]))
                if pair not in processed:
                    rate = self.rng_link.uniform(cfg.B_MIN, cfg.B_MAX)
                    processed.add(pair)
                else:
                    m_neighbors = self.neighbor_matrix[m]
                    m_idx = np.where(m_neighbors == n)[0]
                    rate = (lrm[m, m_idx[0]] if len(m_idx) > 0
                            else self.rng_link.uniform(cfg.B_MIN, cfg.B_MAX))
                lrm[n, idx] = rate
        return lrm

    def _init_prop_delays(self) -> np.ndarray:
        """根据轨道几何计算传播时延。"""
        cfg = self.cfg
        R, c = cfg.ORBIT_RADIUS, cfg.SPEED_OF_LIGHT
        d_intra = 2 * R * np.sin(np.pi / cfg.N_SATS_PER_PLANE)
        d_inter = 2 * R * np.sin(np.pi / cfg.N_PLANES)
        pdm = np.zeros((cfg.N_SATS, 4), dtype=np.float64)
        for n in range(cfg.N_SATS):
            pdm[n] = [d_intra/c, d_intra/c, d_inter/c, d_inter/c]
        return pdm

    def _precompute_solar(self, t_total: int) -> np.ndarray:
        """
        预计算所有卫星在所有时隙的太阳能功率。

        Returns
        -------
        np.ndarray : shape=(N_SATS, T_TOTAL)，float32
        """
        cfg    = self.cfg
        N      = cfg.N_SATS
        omega  = 2 * np.pi / cfg.ORBIT_PERIOD
        t_arr  = np.arange(t_total, dtype=np.float64)
        phases = 2 * np.pi * np.arange(N) / N
        cos_v  = np.cos(omega * t_arr[np.newaxis, :] + phases[:, np.newaxis])
        thr    = np.cos(cfg.LIGHT_RATIO * np.pi)
        is_sun = cos_v > thr
        solar  = np.where(is_sun, cfg.P_SOLAR_MAX * np.maximum(cos_v, 0.0), 0.0)
        return solar.astype(np.float32)

    def _create_satellites(self) -> List[Satellite]:
        cfg  = self.cfg
        sats = []
        for n in range(cfg.N_SATS):
            neighbors   = self.neighbor_matrix[n].tolist()
            link_rates  = {neighbors[i]: float(self.link_rate_matrix[n, i]) for i in range(4)}
            prop_delays = {neighbors[i]: float(self.prop_delay_matrix[n, i]) for i in range(4)}
            sats.append(Satellite(
                sat_id=n, neighbors=neighbors,
                link_rates=link_rates, prop_delays=prop_delays,
                solar_seq=self.solar_seq[n], config=cfg,
            ))
        return sats

    # ── 重置 ──────────────────────────────────────────────────
    def reset(self, seeds: Optional[Dict[str, int]] = None) -> None:
        """
        重置星座到初始状态。

        Parameters
        ----------
        seeds : {'task': int, 'task_param': int, 'dod_init': int}
        """
        self._task_counter.clear()
        if seeds is not None:
            if 'task'       in seeds: self.rng_task       = np.random.default_rng(seeds['task'])
            if 'task_param' in seeds: self.rng_task_param = np.random.default_rng(seeds['task_param'])
            if 'dod_init'   in seeds: self.rng_dod        = np.random.default_rng(seeds['dod_init'])
        for sat in self.satellites:
            sat.reset(rng=self.rng_dod)

    # ── 任务生成与投递 ────────────────────────────────────────
    def generate_tasks(self, t: int) -> Dict[int, List[Task]]:
        """
        生成 t 时隙所有卫星的到达任务（泊松分布）。

        Parameters
        ----------
        t : 当前时隙编号

        Returns
        -------
        {sat_id: [Task, ...]}
        """
        cfg = self.cfg
        tasks_by_sat: Dict[int, List[Task]] = {n: [] for n in range(cfg.N_SATS)}
        for sat_id in range(cfg.N_SATS):
            lam        = cfg.LAMBDA_HIGH if sat_id in self.high_load_sats else cfg.LAMBDA_LOW
            n_arrivals = int(self.rng_task.poisson(lam))
            for _ in range(n_arrivals):
                size       = float(self.rng_task_param.uniform(cfg.S_MIN, cfg.S_MAX))
                cpu_cycles = float(self.rng_task_param.uniform(cfg.H_MIN, cfg.H_MAX))
                deadline   = float(self.rng_task_param.uniform(cfg.D_MAX_MIN, cfg.D_MAX_MAX))
                ck  = (t, sat_id)
                idx = self._task_counter.get(ck, 0)
                self._task_counter[ck] = idx + 1
                tasks_by_sat[sat_id].append(Task(
                    task_id=(t, sat_id, idx), size=size, cpu_cycles=cpu_cycles,
                    deadline=deadline, arrive_slot=t, access_sat=sat_id, tau=cfg.TAU,
                ))
        return tasks_by_sat

    def exchange_info(self) -> Dict[int, Dict]:
        """返回所有卫星的 get_info() 字典（每时隙广播一次）。"""
        return {sat.sat_id: sat.get_info() for sat in self.satellites}

    def deliver_tasks(self, forwarded_tasks: List[Tuple]) -> Dict[int, List[Task]]:
        """
        将转发任务按目标卫星分组。

        Parameters
        ----------
        forwarded_tasks : [(src_id, tgt_id, task), ...]

        Returns
        -------
        {tgt_id: [Task, ...]}
        """
        tasks_by_target: Dict[int, List[Task]] = {n: [] for n in range(self.cfg.N_SATS)}
        for src_id, tgt_id, task in forwarded_tasks:
            tasks_by_target[tgt_id].append(task)
        return tasks_by_target

    def get_neighbor_info(self, sat_id: int) -> Dict[int, Dict]:
        """返回卫星 sat_id 所有邻居的 get_info() 字典。"""
        sat = self.satellites[sat_id]
        return {nid: self.satellites[nid].get_info() for nid in sat.neighbors}

    # ── 统计查询 ──────────────────────────────────────────────
    def get_all_dod(self) -> np.ndarray:
        """返回所有卫星当前 DoD 的数组。"""
        return np.array([sat.dod for sat in self.satellites])

    def get_health_stats(self) -> Dict:
        """返回本时隙电池健康损失统计。"""
        losses     = [sat.slot_health_loss   for sat in self.satellites]
        l_comps    = [sat.slot_delta_l_comp  for sat in self.satellites]
        l_trans    = [sat.slot_delta_l_trans for sat in self.satellites]
        return {
            'total_health_loss': float(sum(losses)),
            'avg_health_loss':   float(sum(losses)  / self.cfg.N_SATS),
            'avg_delta_l_comp':  float(sum(l_comps) / self.cfg.N_SATS),
            'avg_delta_l_trans': float(sum(l_trans) / self.cfg.N_SATS),
        }

    def get_stats(self) -> Dict:
        """返回当前时隙的综合统计指标。"""
        dods     = self.get_all_dod()
        qf_sizes = np.array([sat.qf_size for sat in self.satellites])
        qb_sizes = np.array([sat.qb_size for sat in self.satellites])
        zs       = np.array([sat.z       for sat in self.satellites])
        freqs    = np.array([sat.last_cpu_freq for sat in self.satellites])
        stats    = {
            'avg_dod':     float(np.mean(dods)),
            'max_dod':     float(np.max(dods)),
            'std_dod':     float(np.std(dods)),
            'avg_qf_size': float(np.mean(qf_sizes)),
            'avg_qb_size': float(np.mean(qb_sizes)),
            'avg_z':       float(np.mean(zs)),
            'max_z':       float(np.max(zs)),
            'avg_cpu_freq': float(np.mean(freqs)),
            'max_cpu_freq': float(np.max(freqs)),
        }
        stats.update(self.get_health_stats())
        return stats
