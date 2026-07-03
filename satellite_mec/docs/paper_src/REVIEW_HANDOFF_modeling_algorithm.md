# 交接：系统建模 & 算法设计 重点审查（handoff for new session）

> 面向"从系统建模和算法设计重点审查 demo3.tex"的新会话。目标：在**代码对照**下审查 §4 系统建模 / §5 算法框架的正确性与可辩护性。

## 0. 审查方法（最重要）
**逐条把 demo3.tex 的公式 ↔ `core/` 代码对照。** 本项目历史上"论文落后于代码"，DVFS、奖励塑形等不一致都是靠对照挖出来的——不对照必漏。

## 1. 必读文件
- 论文：`docs/paper_src/demo3.tex`（§4 系统建模 line~176、§5 算法框架 line~678、理论 line~1025）
- 代码：`core/lyapunov.py`（边费用/drift）、`core/dvfs.py`（f_cmp 闭式解+f_floor）、`core/satellite.py`（DoD/能耗更新、act_one 顺序决策）、`core/env.py`（奖励组装：224 完成奖励、248–258 outcome 塑形）、`core/config.py`（定稿参数）
- 参考：`satellite_mec/METHODS.md`
- 本轮修订记录：`docs/paper_src/method_stage0_revisions.md`、`sim_setup_latex.md`、`sim_results_latex.md`

## 2. 已修（本轮完成，勿重复劳动）
- 计算模型：定频公平份额 → **Lyapunov-DVFS**（f_cmp 闭式解 + f_floor 死线下限）
- 奖励：**版本A**（基础 Lyapunov 边费用 + outcome 辅助塑形 r_shape，权重 10/5/5/2/0.05）
- 符号：`Q^{B,cyc}`（周期口径，与队列 bits 区分）、`V_f`、`g^i/w^i`（f_floor）均已入符号表
- `λ_GAE` 0.9 → 0.95（与代码一致）
- eq:delta_dod_comp_task 加"每任务保守代理"说明（定频代理 vs DVFS 实际）
- 理论注：outcome 塑形偏离并入 ε_gap，定理1–3 仍成立
- act_one：§5.3.2 确有"顺序决策"描述（按 d_remain 升序、临时状态更新）

## 3. 高风险待审点（系统建模 + 算法，按优先级）
1. **ε_gap 假设（假设2）= 理论卖点命门**。"隐式逼近理论最优解"全靠假设2成立。审：学习型 MAPPO 策略能否被合理假设在每时隙子问题最优的 ε_gap 内？可辩护性/是否需弱化措辞。
2. **奖励归一化映射未推导**。raw 边费用 c^local（eq:drift_penalty）→ 归一化奖励 c̄^local（eq:cost_local，用 1/(1+V)、V/(1+V)、η/(Z_max·ΔDoD_max)）。审：该归一化是否保持 argmin（不改变最优动作）？若不保持，定理链受影响。
3. **η 的归属不一致**。eq:lyapunov 的 Z² 系数是 **1**，但实验/METHODS 用 **η=0.5** 权重 z_n²；η 目前只出现在奖励归一化里。审：η 到底在 Lyapunov 函数、还是仅奖励超参？二者需自洽。
4. **DVFS vs 奖励代理（c_n² vs f_cmp³）**。本轮用"保守代理"一句话兜住（式\eqref{eq:delta_dod_comp_task}）。严格审可能要求奖励的 DoD 项与 DVFS 一致，或给更强论证。
5. **定理 0–3 证明未展开**。drift-plus-penalty 上界推导、B(t) 常数（B^F/B^B/eq:B_t 的 −ΣQ^B·r 与 −ΣZ·ΔDoD^solar 项）需逐步核对是否成立。
6. **一阶泰勒线性化误差**。eq:delta_L_comp 在 DoD_n(t) 处展开；若单任务处理周期内 DoD 变化较大，线性化误差不可忽略。审"小增量"假设是否成立、是否需误差界。
7. **est_comp_delay 用 c_m（满频）**。eq:est_comp_delay 用 c_m 估邻居计算时延；DVFS 下实际 f_cmp≤c_m，截止可行性检查偏乐观（靠 f_floor 兜？）。审一致性。
8. **δ_max 措辞**。eq:delta_max"第一项对应 N^B=1"在 DVFS 下应改为"f_cmp 满频 c_n 时"（界值 τκc_n³ 仍对）。
9. **准入阈值 θ_n**（eq:theta，源自 MHSPO）在电池感知设定下是否仍恰当。

## 4. 诚实红线（沿用）
Pareto 非综合最优；数字只从 `multiseed_lh4.json`/`diagnostics_lh4.json` 取；引用不编造（`bibitem{2}–{6}` 待补真实信息）；N=192 为投影需标注（设置节 `%TODO` 仍搁置）。

## 5. 建议给新会话的开场
"审查 demo3.tex 的系统建模(§4)与算法设计(§5)。**先逐式对照 core/ 代码**，重点核第 3 节的 9 个高风险点，尤其 ε_gap 假设、奖励归一化映射、η 归属、定理证明。守诚实红线。"
