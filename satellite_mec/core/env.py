"""
core/env.py
===========
卫星MEC仿真环境（实现 EnvInterface）。

使用示例
--------
    from core import Config, SatelliteMECEnv

    cfg = Config()
    env = SatelliteMECEnv(cfg)
    env.reset(phase='eval', seeds={'task': 42, 'task_param': 43, 'dod_init': 44})

    for t in range(5400):
        next_obs, rewards, done, info = env.step(policy=my_policy)
        print(f"slot={t}, CR={info['completion_rate']:.3f}")

    print(f"Eval CR: {env.get_eval_completion_rate():.3f}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from core.constellation import Constellation
from core.lyapunov import LyapunovCalculator
from interfaces import EnvInterface

if TYPE_CHECKING:
    from core.config import Config
    from interfaces import PolicyInterface


class SatelliteMECEnv(EnvInterface):
    """
    卫星MEC多智能体仿真环境。

    每个时隙的执行流程
    ------------------
    1. exchange_info()      — 广播全局状态
    2. update_solar()       — 更新日照状态
    3. generate_tasks()     — 泊松任务到达
    4. admit_task()         — 入队（超容量丢弃）
    5. init_temp_state()    — 初始化预测状态
    6. remove_timeout_tasks() — 清理超时任务
    7. sort_forward_queue() — 按紧迫性排序
    8. get_actions()        — 策略决策
    9. apply_action()       — 执行动作（本地/转发）
    10. process_tasks()     — 推进计算
    11. deliver_tasks()     — 转发投递
    12. update_queues()     — 更新队列
    13. update_dod()        — 更新 DoD
    14. update_alpha_avg()  — 更新滑动均值
    """

    def __init__(self, config: "Config", lyapunov_calc: Optional[LyapunovCalculator] = None):
        self.cfg            = config
        self.constellation  = Constellation(config)
        self.lyapunov_calc  = lyapunov_calc or LyapunovCalculator(config)
        self.current_slot:  int  = 0
        self.phase:         str  = 'train'
        self.slots_in_episode: int = 0
        self._global_info:  Dict = {}

        # 累计统计
        self.episode_arrived: int = 0
        self.episode_done:    int = 0
        self.episode_timeout: int = 0
        self._prev_episode_timeout: int = 0

        # 评估阶段统计
        self.eval_arrived:     int   = 0
        self.eval_done:        int   = 0
        self.eval_timeout:     int   = 0
        self.eval_satisfied:   int   = 0
        self.eval_satisfaction_denom: int = 0

    # ── 重置 ──────────────────────────────────────────────────
    def reset(self, phase: str = 'train',
              seeds: Optional[Dict[str, int]] = None) -> Dict:
        self.phase             = phase
        self.current_slot      = 0
        self.slots_in_episode  = 0
        self.constellation.reset(seeds=seeds)
        self.episode_arrived   = 0
        self.episode_done      = 0
        self.episode_timeout   = 0
        self._prev_episode_timeout = 0
        if phase == 'eval':
            self.eval_arrived            = 0
            self.eval_done               = 0
            self.eval_timeout            = 0
            self.eval_satisfied          = 0
            self.eval_satisfaction_denom = 0
        self._global_info = self.constellation.exchange_info()
        return {}

    # ── 主循环 ────────────────────────────────────────────────
    def step(self, actions: Optional[Dict[int, List[int]]] = None,
             policy: Optional["PolicyInterface"] = None
             ) -> Tuple[Dict, Dict, bool, Dict]:
        t   = self.current_slot
        cfg = self.cfg
        sats = self.constellation.satellites

        # 1. 广播全局状态 & 更新日照
        self._global_info = self.constellation.exchange_info()
        for sat in sats:
            sat.update_solar(t)

        # 2. 任务生成与入队
        tasks_by_sat  = self.constellation.generate_tasks(t)
        slot_arrived  = sum(len(v) for v in tasks_by_sat.values())
        self.episode_arrived += slot_arrived
        rejected_count = 0
        rejected_per_sat = {n: 0 for n in range(cfg.N_SATS)}
        for sat in sats:
            for task in tasks_by_sat.get(sat.sat_id, []):
                if not sat.admit_task(task):
                    rejected_count += 1
                    rejected_per_sat[sat.sat_id] += 1

        # 3. 策略分支：policy 对象 or 预计算 actions
        sequential = (policy is not None and hasattr(policy, 'act_one'))
        if policy is not None:
            for sat in sats:
                sat.init_temp_state()
                slot_to = sat.remove_timeout_tasks(t)
                self.episode_timeout += len(slot_to)
                sat.sort_forward_queue(t)
            if hasattr(policy, 'collect_critic_values'):
                policy.collect_critic_values(self)
            if not sequential:
                obs     = self.get_observations()
                masks   = self.get_action_masks()
                actions = policy.get_actions(obs, masks)

        if actions is None:
            actions = {}

        # 4. 执行动作（sequential：每个 task 决策时构造最新 state；batch：用预计算 actions）
        next_obs   = {n: [] for n in range(cfg.N_SATS)}
        rewards    = {n: 0.0 for n in range(cfg.N_SATS)}
        n_executed = {n: 0   for n in range(cfg.N_SATS)}
        all_forwarded: List[Tuple] = []

        for sat in sats:
            n = sat.sat_id
            neighbor_info = {nid: self._global_info[nid] for nid in sat.neighbors}
            if policy is None:
                sat.init_temp_state()
                slot_timeout = sat.remove_timeout_tasks(t)
                sat.sort_forward_queue(t)
                self.episode_timeout += len(slot_timeout)

            sat_actions     = actions.get(n, [])
            task_action_idx = 0
            tasks_to_process = list(sat.forward_queue)

            for task in tasks_to_process:
                neighbor_nb = {nid: self._global_info[nid]['nb'] for nid in sat.neighbors}
                mask = sat.get_action_mask(task, t, neighbor_nb)
                if mask.sum() == 0:
                    continue
                state = sat.get_state(task, t, neighbor_info)
                next_obs[n].append(state)

                if sequential:
                    action, log_prob = policy.act_one(state, mask)
                elif task_action_idx < len(sat_actions):
                    action = sat_actions[task_action_idx]; task_action_idx += 1
                    log_prob = 0.0
                else:
                    action = int(np.argmax(mask))
                    log_prob = 0.0
                # 策略明确要求 local (=0)，但本地容量满时保留在队列等下一时隙
                if action == 0 and mask[0] == 0:
                    continue
                if action >= len(mask) or mask[action] == 0:
                    action = int(np.argmax(mask))

                reward, forward_info = sat.apply_action(
                    task, action, t, neighbor_info, self.lyapunov_calc)
                rewards[n]    += reward
                n_executed[n] += 1
                if sequential and hasattr(policy, 'record_task_transition'):
                    policy.record_task_transition(
                        sat_id=n, slot_t=t, state=state, action=action,
                        log_prob=log_prob, mask=mask, task_reward=reward,
                    )
                if forward_info is not None:
                    tgt_id, fwd_task = forward_info
                    all_forwarded.append((n, tgt_id, fwd_task))

        # 5. 计算推进
        slot_done       = 0
        slot_satisfied  = 0
        slot_e2e_delays: List[float] = []
        slot_done_deadlines: List[float] = []          # 完成任务的 ddl，用于 slack ratio
        slot_timeout_deadlines: List[float] = []       # 超时任务的 ddl，用于"全任务"延迟 PDF
        nb_start_map    = {sat.sat_id: sat.nb for sat in sats}
        done_per_sat    = {n: 0 for n in range(cfg.N_SATS)}
        timeout_per_sat = {n: 0 for n in range(cfg.N_SATS)}
        satisfied_per_sat = {n: 0 for n in range(cfg.N_SATS)}
        delay_sum_per_sat = {n: 0.0 for n in range(cfg.N_SATS)}   # Zhong Y_n: per-sat E2E 时延和
        delay_cnt_per_sat = {n: 0 for n in range(cfg.N_SATS)}

        for sat in sats:
            done_tasks, compute_timeout = sat.process_tasks(t)
            slot_done += len(done_tasks)
            done_per_sat[sat.sat_id] += len(done_tasks)
            if compute_timeout:
                self.episode_timeout += len(compute_timeout)
                timeout_per_sat[sat.sat_id] += len(compute_timeout)
                # 记录超时任务的 deadline，供 "all admitted" 延迟 PDF 使用
                slot_timeout_deadlines.extend(t.deadline for t in compute_timeout)
            for task in done_tasks:
                real_delay = (task.finish_slot - task.arrive_slot) * cfg.TAU
                slot_e2e_delays.append(real_delay)
                slot_done_deadlines.append(task.deadline)
                delay_sum_per_sat[sat.sat_id] += real_delay
                delay_cnt_per_sat[sat.sat_id] += 1
                if real_delay <= task.deadline:
                    slot_satisfied += 1
                    satisfied_per_sat[sat.sat_id] += 1
                    rewards[sat.sat_id] += cfg.COMPLETION_BONUS

        self.episode_done += slot_done
        slot_timeout_count = self.episode_timeout - self._prev_episode_timeout
        self._prev_episode_timeout = self.episode_timeout
        satisfaction_denom      = slot_done + slot_timeout_count
        slot_satisfaction_rate  = slot_satisfied / max(satisfaction_denom, 1)
        slot_satisfaction_orig  = slot_satisfied / max(slot_done, 1)

        # 6. 转发投递
        tasks_by_target = self.constellation.deliver_tasks(all_forwarded)
        for sat in sats:
            received = tasks_by_target.get(sat.sat_id, [])
            transit_timeout = sat.update_queues(
                new_arrivals=[], received_tasks=received, current_slot=t)
            if transit_timeout:
                self.episode_timeout += len(transit_timeout)
                timeout_per_sat[sat.sat_id] += len(transit_timeout)

        # 7. DoD / alpha 更新
        for sat in sats:
            sat.update_dod(nb_start=nb_start_map[sat.sat_id])
            sat.update_alpha_avg()

        # 8. Outcome-aware reward 注入 + reward ledger 记录组成
        #    ledger 总是计算（包括 eval 阶段），方便诊断；reward 注入仅训练阶段生效
        ledger = {'done': 0.0, 'timeout': 0.0, 'reject': 0.0, 'hl': 0.0, 'queue': 0.0, 'dod': 0.0}
        for sat in sats:
            n = sat.sat_id
            queue_pressure = (sat.qf_size + sat.qb_size) / max(cfg.QUEUE_NORM, 1.0)
            r_done    =   cfg.W_DONE    * satisfied_per_sat[n]
            r_timeout = - cfg.W_TIMEOUT * timeout_per_sat[n]
            r_reject  = - cfg.W_REJECT  * rejected_per_sat[n]
            r_hl      = - cfg.W_HL      * sat.slot_health_loss / max(cfg.HL_NORM, 1e-12)
            r_queue   = - cfg.W_QUEUE   * queue_pressure
            r_dod     = - cfg.W_DOD     * sat.dod        # MADRL-DoD: 罚 DoD 存量 δ（默认 W_DOD=0 无影响）
            if self.phase == 'train':
                rewards[n] += r_done + r_timeout + r_reject + r_hl + r_queue + r_dod
            ledger['done']    += r_done
            ledger['timeout'] += r_timeout
            ledger['reject']  += r_reject
            ledger['hl']      += r_hl
            ledger['queue']   += r_queue
            ledger['dod']     += r_dod
        # 已经在 step 内累加进 rewards 的"action_cost"（来自 sat.apply_action 返回的 −Lyapunov_cost）
        # 这里再做一次汇总，避免重复计算时把 reward 整体丢失
        ledger['action_cost'] = float(sum(rewards.values())) - sum(ledger.values()) \
                                if self.phase == 'train' else float(sum(rewards.values()))
        ledger['total'] = float(sum(rewards.values()))

        self.current_slot     += 1
        self.slots_in_episode += 1
        done = (self.slots_in_episode >= cfg.K_ROLLOUT)
        if done:
            self.slots_in_episode = 0

        stats = self.constellation.get_stats()
        info  = {
            'slot': t, 'phase': self.phase,
            'arrived': slot_arrived, 'done_tasks': slot_done,
            'slot_timeout': slot_timeout_count,
            'episode_timeout': self.episode_timeout,
            'rejected': rejected_count, 'forwarded': len(all_forwarded),
            'n_executed': n_executed,
            'avg_dod':  stats['avg_dod'],  'max_dod': stats['max_dod'],
            'std_dod':  stats['std_dod'],
            'avg_qf_size': stats['avg_qf_size'], 'avg_qb_size': stats['avg_qb_size'],
            'avg_z': stats['avg_z'], 'max_z': stats['max_z'],
            'avg_health_loss':   stats.get('avg_health_loss', 0.0),
            'avg_delta_l_comp':  stats.get('avg_delta_l_comp', 0.0),
            'avg_delta_l_trans': stats.get('avg_delta_l_trans', 0.0),
            'avg_cpu_freq':      stats.get('avg_cpu_freq', 0.0),
            'max_cpu_freq':      stats.get('max_cpu_freq', 0.0),
            'slot_system_energy':       stats.get('slot_system_energy', 0.0),       # J/槽，系统总能耗
            'slot_system_energy_comp':  stats.get('slot_system_energy_comp', 0.0),
            'slot_system_energy_trans': stats.get('slot_system_energy_trans', 0.0),
            'total_queue_size':  stats['avg_qf_size'] + stats['avg_qb_size'],       # 字节口径(MB)
            'queue_task_count':  stats.get('avg_queue_tasks', 0.0),                 # 任务个数口径(per-sat)
            'qf_task_count':     stats.get('avg_qf_tasks', 0.0),
            'qb_task_count':     stats.get('avg_qb_tasks', 0.0),
            'reward_ledger':     ledger,    # 诊断：每 slot reward 组成
            'per_sat_dod':      [sat.dod for sat in self.constellation.satellites],
            # Zhong 忠实 reward 所需 per-sat 量（加法式，不影响既有逻辑）
            'per_sat_queue':    [s.qf_size + s.qb_size for s in self.constellation.satellites],
            'per_sat_energy':   [getattr(s, 'slot_comp_energy', 0.0) + getattr(s, 'slot_trans_energy', 0.0)
                                 for s in self.constellation.satellites],
            'per_sat_delay_sum':[delay_sum_per_sat[s.sat_id] for s in self.constellation.satellites],
            'per_sat_delay_cnt':[delay_cnt_per_sat[s.sat_id] for s in self.constellation.satellites],
            'episode_arrived':   self.episode_arrived,
            'episode_done':      self.episode_done,
            'episode_timeout':   self.episode_timeout,
            'completion_rate':   self.episode_done / max(self.episode_arrived, 1),
            'slot_satisfied':           slot_satisfied,
            'slot_satisfaction_rate':   slot_satisfaction_rate,
            'slot_satisfaction_rate_orig': slot_satisfaction_orig,
            'slot_e2e_delays':          slot_e2e_delays,
            'slot_done_deadlines':      slot_done_deadlines,
            'slot_timeout_deadlines':   slot_timeout_deadlines,
        }
        if self.phase == 'eval':
            self.eval_arrived            += slot_arrived
            self.eval_done               += slot_done
            self.eval_timeout            += slot_timeout_count   # 修：之前累加 episode 累计值
            self.eval_satisfied          += slot_satisfied
            self.eval_satisfaction_denom += satisfaction_denom
            info['eval_completion_rate']       = self.eval_done / max(self.eval_arrived, 1)
            info['eval_satisfaction_rate']     = (self.eval_satisfied
                                                  / max(self.eval_satisfaction_denom, 1))
            info['eval_satisfaction_rate_orig'] = (self.eval_satisfied
                                                   / max(self.eval_done, 1))
        return next_obs, rewards, done, info

    # ── 观测接口 ──────────────────────────────────────────────
    def get_critic_obs(self) -> Dict[int, np.ndarray]:
        t = self.current_slot
        # 方案 B：每次构造 critic 时同步算一次全局摘要（所有卫星共享）
        global_summary = self.constellation.get_global_summary(t)
        return {
            sat.sat_id: sat.get_critic_state(
                {nid: self._global_info.get(nid, {}) for nid in sat.neighbors},
                t, global_summary=global_summary)
            for sat in self.constellation.satellites
        }

    def get_action_masks(self) -> Dict[int, List[np.ndarray]]:
        t = self.current_slot
        masks = {}
        for sat in self.constellation.satellites:
            n = sat.sat_id
            neighbor_nb = {nid: self._global_info.get(nid, {}).get('nb', 0)
                           for nid in sat.neighbors}
            masks[n] = [sat.get_action_mask(task, t, neighbor_nb)
                        for task in sat.forward_queue if not task.is_timeout(t)]
        return masks

    def get_observations(self) -> Dict[int, List[np.ndarray]]:
        t = self.current_slot
        obs = {}
        for sat in self.constellation.satellites:
            n = sat.sat_id
            neighbor_info = {nid: self._global_info.get(nid, {}) for nid in sat.neighbors}
            obs[n] = [sat.get_state(task, t, neighbor_info)
                      for task in sat.forward_queue if not task.is_timeout(t)]
        return obs

    def get_completion_rate(self) -> float:
        return self.episode_done / max(self.episode_arrived, 1)

    def get_eval_completion_rate(self) -> float:
        return self.eval_done / max(self.eval_arrived, 1)

    # 便捷属性（供外部查询维度）
    def get_state_dim(self)        -> int: return self.cfg.get_state_dim()
    def get_critic_state_dim(self) -> int: return self.cfg.get_critic_state_dim()
    def get_action_dim(self)       -> int: return self.cfg.get_action_dim()
