# REVISION_PLAN.md — demo3_fin2.tex 投稿前结构性重写计划

> 目标：把论文从"一个 MAPPO 变体"重定位为 **Battery Aging-aware Task Offloading Framework**，
> 其核心算法为 **Battery Aging-aware Policy Learning Mechanism**，**参数共享 MAPPO 只是底层优化器**。
> 约束：不改实现/数学模型/实验设置/实验数据；不新增实验、消融、定理；只重写叙事与结构。

## 1. 当前论文的定位问题
- 现稿（提交 c354c94）已把方法命名为 **DA-MAPPO**，仍带"MAPPO 品牌"，容易被读成"MAPPO 小改"。
- 术语混用：以"寿命损伤"为主，夹带"寿命/老化/损耗"，未统一到 **电池老化（battery aging）**。
- 三层关系（Framework / Policy Learning / MAPPO Optimizer）未显式区分；MAPPO 与 PPO/GAE 混在方法主体里。
- Introduction 未把 **"能耗 ≠ 电池老化"** 作为有记忆点的核心命题突出。
- Related Work 未显式分成三类（SEC 卸载 / 能量与电池感知 / MARL 卸载）。
- 复杂度虽已四分类，但个别措辞仍可更严谨；实验分析主语偏"DA-MAPPO/MAPPO"。

## 2. 新的三层定位（全文统一）
| 层 | 名称 | 内容 |
|---|---|---|
| L1 系统 | Battery Aging-aware Task Offloading Framework | 面向 SEC 的长期电池老化感知卸载问题与框架 |
| L2 算法 | Battery Aging-aware Policy Learning Mechanism | aging-aware 状态/信号、结构化稠密奖励、可行性感知掩码、一跳分布式策略、参数共享、CTDE |
| L3 优化器 | Parameter-sharing MAPPO Optimizer | 标准 MAPPO backbone（Actor/Critic/PPO-clip/GAE），仅作策略优化器 |

- **方法名**：保留 `DA-MAPPO` 作为**整个框架**的稳定简称（明确其指代 Framework 而非 PPO），实验图/表**标签暂不改**（用户指示）。
- 禁止表述："propose a novel MAPPO/PPO"、"improve PPO"、"new MAPPO variant"、全局最优、严格队列稳定性定理、复杂度与规模无关、精确预测绝对寿命、"Lyapunov optimization algorithm"、无据"首次"。

## 3. 各章节修改计划
1. **摘要**：背景→能耗优化不足→"能耗≠电池老化"洞察→提出 framework→policy learning 机制组件→MAPPO 作优化器→现有结果。前两句不出现 MAPPO。
2. **引言**：7 段结构，突出核心命题"equal energy ≠ equal battery aging"；DoD 非线性、能耗集中于高 DoD 卫星的危害；引出 framework+机制。
3. **贡献**：三条——(C1) 问题与建模；(C2) Battery Aging-aware Policy Learning 机制；(C3) 分布式可扩展执行（实验验证附于 C3 末，不单列第四条）。
4. **相关工作**：显式三类——SEC 与任务卸载 / 能量感知与电池感知卫星组网 / MARL 卸载；克制的 gap 表述（"limited work"，不用"no work/first"）。
5. **问题形式化**：目标定位为最小化长期电池老化（受时延/队列/资源/可行性约束）；补一段"与能耗最小化的区别：老化通过非线性 DoD 关系给能耗赋状态相关代价"。
6. **算法章重排为 IV.A–I**：A 框架总览(三层)/B 老化感知状态/C 可行性感知动作与掩码/D 结构化稠密奖励/E Lyapunov-inspired 队列引导/F 电池老化感知多智能体策略/G 参数共享 MAPPO 优化器/H 训练与分布式执行/I 复杂度。内容基本沿用现有段落，仅调标题、衔接与总览；MAPPO 收进 G 并声明为 standard backbone。
7. **Lyapunov**：保留 inspired，降低理论宣称（"受 Lyapunov 队列压力均衡原理启发"），不写 Lyapunov 算法/不宣称稳定性定理/最优界。
8. **复杂度**：区分 training（随智能体数/联合维度/critic 输入增长）与 execution（仅取决于有界局部观测维度），不写"与规模无关"。
9. **实验分析**：不改数据；主语改"所提框架/电池老化感知策略 enables…"；每图回答 老化/任务性能/队列/掩码作用；无消融支撑的因果改为"与…设计意图一致"。
10. **结论**：重申三层；Future work（电化学模型/真实轨迹/标定/异构电池/细粒度信用分配/在线自适应），不宣称已解决。
11. **术语统一**：`寿命损伤/损耗`→**电池老化**（cost 语境用"老化代价/相对老化代价"）；首现定义"本文以所采用 DoD--循环寿命关系量化电池老化，表示相对寿命退化而非绝对剩余寿命"。

## 4. 高风险表述清单（逐条软化，见正文修改）
guarantees queue stability / global optimum / independent of constellation size / accurately predicts lifetime /
Lyapunov optimization solves / first work / propose novel MAPPO —— 全部按 brief §13 改写或删除。

## 5. 明确不改动的内容
- 所有公式的数学实质、Actor/Critic/PPO/GAE 更新规则、网络结构；
- 系统模型的方程与变量符号（仅统一文字术语）；
- 实验设置、图表数据、数值结果、消融变体名（含 `LyapunovGreedy`）；
- 不新增实验/消融/定理/网络/信用分配模块；不生成新图（用户指示后补）。
