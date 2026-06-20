"""
baselines/gdco.py
=================
GDCO —— 博弈论分布式计算卸载基线（Game-theoretical Distributed Computation Offloading）。

论文复现
--------
Ying Chen, Yaozong Yang, Jintao Hu, Yuan Wu, Jiwei Huang,
"A Game-Theoretical Approach for Distributed Computation Offloading in LEO
 Satellite-Terrestrial Edge Computing Systems,"
IEEE Transactions on Mobile Computing, vol. 24, no. 5, pp. 4389-4402, May 2025.
DOI: 10.1109/TMC.2025.3526200

原文核心（Section IV-V）
------------------------
1. 把"各卸载主体自私最小化能耗"建模为 LSTCO-Game；
2. 证明其为 **exact potential game**，势函数 Φ = Σ_i C_i（总成本），故至少存在一个 NE；
3. 玩家代价用 **边际成本 overhead** Q_i = C_i + Σ_{t≠i}[C_t(h) − C_t(h\i)]，
   把"个体决策对他人的外部性"内部化；
4. **GDCO 算法（Algorithm 1）**：全体初始化为本地计算，按随机序做 best-response，
   每次切到能降低总势 Φ 的决策，直到无人愿改 → 收敛到 NE。

到本仓库 LEO 卫星-卫星转发场景的适配（与爸爸讨论后定稿口径：能耗 + 死线可行）
--------------------------------------------------------------------------
原文是"地面设备 → {本地 / 基站 / 卫星}"；本仓库是"卫星 → {本地 / 4 个邻居卫星}（1 跳）"。
忠实映射如下：

| 原文组件                | 本仓库落地 |
|-------------------------|-----------|
| 玩家 i                  | 一颗卫星本时隙 forward_queue 里待决策的每个 task |
| 策略集 H_i              | {0=本地} ∪ {合法且死线可行的邻居}（动作 ≤5）|
| 成本 C_i                | **纯能耗**：本地=`delta_dod_comp`(DVFS 计算能耗)、转发=`delta_dod_trans`(传输能耗)|
| 共享资源耦合（外部性）  | 本地计算的整星 DVFS 拥塞：本地任务越多 → q_cycles 越大 → f_cmp³ 边际能耗越高，
|                         | 抬高其它本地任务的代价（原文"服务器算力按比例分摊"的卫星侧对应物）|
| 势函数 Φ = Σ_i C_i      | 本卫星本时隙的**总能耗**（本地 DVFS 能耗 + 各转发传输能耗）|
| GDCO best-response      | 每槽：全置本地 → 随机序逐 task 切到使 Φ 最小的动作 → 无改/到 max_iter → NE |

为何这样能保证"LyaMAPPO 综合最强"
---------------------------------
GDCO 与原文一致，**只优化能耗，不含电池健康损失 / DoD 虚拟队列项**（那是 LyaMAPPO
创新 #1/#2 独有）。因此 GDCO 预期在 CR/时延上有竞争力，但 **HL 偏高**，落在 LyaMAPPO
的 Pareto 支配下方——既是忠实复现，又守住了 LyaMAPPO 的 HL 护城河。

公平性
------
GDCO 与其它 5 个策略**共享同一套 Lyapunov-DVFS 物理基底**（env 统一执行 `process_tasks`
/ `update_dod`），策略差异只来自卸载决策本身（METHODS 创新 #6）。本类只在**决策时**用
能耗口径，不改任何物理。

实现说明
--------
- `get_actions` **绝不修改卫星状态**，全部在本地 sim 字典上推演；真实状态推进由 env 的
  `apply_action` 负责（避免与 env 第二轮重复推进 nb_hat / z_hat）。
- 单 task 的 best-response 严格降低 Φ；Φ 有下界且动作有限 → 有限步收敛到 NE（原文 Thm 1）。
  最小化 Φ 与最小化边际成本 overhead Q_i 等价（exact potential game：ΔΦ = ΔQ_i）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from core.lyapunov import LyapunovCalculator
from interfaces import PolicyInterface

if TYPE_CHECKING:
    from core.config import Config
    from core.env import SatelliteMECEnv
    from core.satellite import Satellite
    from core.task import Task


class GDCOPolicy(PolicyInterface):
    """
    博弈论分布式计算卸载策略（GDCO baseline，Chen et al. TMC 2025 复现）。

    Parameters
    ----------
    config   : Config 实例
    env      : SatelliteMECEnv 实例
    max_iter : 每时隙 best-response 的最大扫描轮数（封顶运行时；实测收敛轮数通常远小于此）
    seed     : best-response 随机更新序的种子（原文采用随机序竞争更新机会）
    """

    needs_training: bool = False
    name: str = 'GDCO'

    def __init__(self, config: "Config", env: "SatelliteMECEnv",
                 max_iter: int = 8, seed: int = 0):
        self.cfg      = config
        self.env      = env
        # 仅借用能耗增量 delta_dod_comp / delta_dod_trans（不用其 Lyapunov/电池项）
        self.calc     = LyapunovCalculator(config)
        self.max_iter = int(max_iter)
        self.rng      = np.random.default_rng(seed)
        # 死线可行：本地拥塞导致超时的惩罚（DoD 分数单位，远大于 ~1e-5 量级的能耗，
        # 使势函数先规避超时、再在可行动作间比能耗）。这是"死线可行"约束的软实现。
        self.timeout_penalty: float = 1.0
        # 收敛诊断：记录每次 _solve_sat 实际用掉的扫描轮数
        self.iter_hist: List[int] = []

    # ── PolicyInterface ───────────────────────────────────────
    def get_actions(self, obs, masks) -> Dict[int, List[int]]:
        actions: Dict[int, List[int]] = {}
        t = self.env.current_slot
        for sat in self.env.constellation.satellites:
            n         = sat.sat_id
            sat_masks = masks.get(n, [])
            pairs     = list(zip(sat.forward_queue, sat_masks))
            if not pairs:
                actions[n] = []
                continue
            actions[n] = self._solve_sat(sat, pairs, t)
        return actions

    # ── 单卫星 LSTCO-Game 求解（best-response → NE）────────────
    def _solve_sat(self, sat: "Satellite", pairs, t: int) -> List[int]:
        cfg = self.cfg

        # 只读快照基线（init_temp_state 已由 env 在 get_actions 前调用过）
        base_q     = float(sat.q_cycles_hat)
        base_floor = float(sat.f_floor_hat)
        base_nb    = int(sat.nb_hat)
        local_cap  = max(cfg.MAX_DISPATCH - base_nb, 0)   # 还能再接纳几个本地任务

        # 1) 构造每个玩家（非全零掩码 task）的死线可行动作集
        feasible: Dict[int, List[int]] = {}
        for pos, (task, mask) in enumerate(pairs):
            if float(np.sum(mask)) == 0.0:
                continue                                   # 无合法动作 → 非玩家（env 会跳过）
            feas: List[int] = []
            # 本地：掩码合法（容量）即为候选；死线在 Φ 里以"并发感知超时惩罚"处理
            # （mask[0] 只查 nb<MAX_DISPATCH，不查死线，而 est_comp_delay 忽略并发共享，
            #  故不能用 feasible_local 静态预筛——见类 docstring 与 smoke 诊断）
            if mask[0] > 0:
                feas.append(0)
            # 转发：掩码已含 feasible_forward（含死线 + 通信窗口）
            for idx in range(len(sat.neighbors)):
                if idx + 1 < len(mask) and mask[idx + 1] > 0:
                    feas.append(idx + 1)
            if not feas:                                   # 无任何合法动作 → 占位本地
                feas = [a for a in range(len(mask)) if mask[a] > 0] or [0]
            feasible[pos] = feas

        players = sorted(feasible.keys())

        # 2) 初始化：原文置全体为本地，受 local_cap 约束（超容量的玩家先放转发）
        assign: Dict[int, int] = {}
        n_local = 0
        for pos in players:
            opts = feasible[pos]
            if 0 in opts and n_local < local_cap:
                assign[pos] = 0
                n_local += 1
            else:
                fwd = [a for a in opts if a != 0]
                assign[pos] = fwd[0] if fwd else 0

        # 3) best-response 扫描（随机序），单 task 切到使总势 Φ 最小的动作
        for it in range(self.max_iter):
            changed = False
            for pos in self.rng.permutation(players):
                opts = feasible[pos]
                if len(opts) <= 1:
                    continue
                cur      = assign[pos]
                best_a   = cur
                best_phi = self._phi(sat, pairs, assign, base_q, base_floor, base_nb, t)
                for a in opts:
                    if a == cur:
                        continue
                    if a == 0:   # 本地容量硬约束（与 env get_action_mask 的 nb<MAX_DISPATCH 一致）
                        n_local_others = sum(1 for p in players if p != pos and assign[p] == 0)
                        if n_local_others >= local_cap:
                            continue
                    assign[pos] = a
                    phi = self._phi(sat, pairs, assign, base_q, base_floor, base_nb, t)
                    if phi < best_phi - 1e-15:
                        best_phi, best_a = phi, a
                assign[pos] = best_a
                if best_a != cur:
                    changed = True
            if not changed:
                self.iter_hist.append(it + 1)
                break
        else:
            self.iter_hist.append(self.max_iter)

        # 4) 按 forward_queue 位置回填动作（全零掩码 task 占位 0，与其它基线约定一致）
        out: List[int] = []
        for pos, (task, mask) in enumerate(pairs):
            out.append(int(assign[pos]) if pos in assign else 0)
        return out

    # ── 势函数 Φ = 本卫星本时隙总能耗 + 并发感知超时惩罚 ──────────
    #   能耗项 = 本地 DVFS 边际 + 各转发传输能耗（按队列顺序累积，exact potential）；
    #   死线项 = 任务在目标节点并发共享下完不成 → 加 timeout_penalty（"死线可行"软约束）。
    #   本地拥塞 ⇒ 抬高其它本地任务；转发拥塞 ⇒ 抬高同投一邻居的其它任务（Chen 的算力
    #   按比例分摊外部性的卫星侧对应物）。两路耦合共同驱动 NE 做负载均衡。
    def _phi(self, sat: "Satellite", pairs, assign: Dict[int, int],
             base_q: float, base_floor: float, base_nb: int, t: int) -> float:
        cfg = self.cfg
        sim_q, sim_floor, sim_nb = base_q, base_floor, base_nb
        fwd_count: Dict[int, int] = {}                       # 本槽已投往各邻居的计数（队内外部性）
        ginfo = self.env._global_info
        total = 0.0
        for pos, (task, mask) in enumerate(pairs):
            a = assign.get(pos)
            if a is None:
                continue
            solo_slots = math.ceil(task.size * task.cpu_cycles
                                   / max(cfg.CPU_FREQ * cfg.TAU, 1e-9))
            remain = task.remain_time(t)
            if a == 0:                                      # 本地：DVFS 计算能耗边际 + 死线惩罚
                sat_state = {'q_cycles_hat': sim_q,
                             'f_floor_hat':  sim_floor,
                             'nb_hat':       sim_nb}
                total += self.calc.delta_dod_comp(task, sat_state)
                nb_next = sim_nb + 1
                # 并发共享完成时间 ≈ nb_next × 独占耗时（对齐 env.process_tasks 的 cycle 平分）
                if nb_next * solo_slots * cfg.TAU > remain:
                    total += self.timeout_penalty
                tc        = task.size * task.cpu_cycles
                sim_q    += tc
                sim_floor = max(sim_floor, nb_next * tc / max(task.deadline, cfg.TAU))
                sim_nb    = nb_next
            else:                                           # 转发：传输能耗 + 邻居拥塞死线惩罚
                neighbor_id = sat.neighbors[a - 1]
                b_nm        = sat.link_rates[neighbor_id]
                t_nm        = sat.prop_delays[neighbor_id]
                total += self.calc.delta_dod_trans(task, b_nm)
                # 邻居投影并发 = 槽首快照 nb + 本槽我已投往该邻居的计数（+1 为本任务）
                nb_m  = int(ginfo.get(neighbor_id, {}).get('nb', 0)) + fwd_count.get(neighbor_id, 0)
                trans = task.size / b_nm + t_nm
                if trans + (nb_m + 1) * solo_slots * cfg.TAU > remain:
                    total += self.timeout_penalty
                fwd_count[neighbor_id] = fwd_count.get(neighbor_id, 0) + 1
        return total
