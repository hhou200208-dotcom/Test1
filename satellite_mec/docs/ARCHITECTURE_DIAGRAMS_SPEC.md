# 架构图构图清单（系统模型图 + LyaMAPPO 框架图）

> 手画 / draw.io / TikZ 都能照此画。参数全部对齐代码真实值（54维/245维/V=50/η=0.5 等）。
> 论文场景：N=192 LEO 卫星（16×12 Walker）。

---

## 图 1：系统模型图（System Model — LEO 卫星 MEC 卸载）

### 整体布局（三区）
左：星座全景 → 中：单星内部放大 → 右：电池能量循环

### 模块清单
| # | 模块 | 画法 | 关键标注 |
|---|---|---|---|
| 1 | LEO 星座 | Walker 网格 16 平面×12 星=192（示意画 4×4，标"…×N"）| `N=192 LEO satellites` |
| 2 | 星间链路 ISL | 每颗连 4 邻居（同轨前/后 + 邻轨左/右），实线 | `ISL: rate R_{n,m}, prop delay τ_{n,m}` |
| 3 | 太阳/地影 | 太阳图标 + 阴影弧（轨道周期日照/地影交替）| `solar P_solar, eclipse cycle` |
| 4 | 高/低负载星 | 5 高负载红、20 低负载蓝（示意）| `λ_high=4.0 / λ_low` |
| 5 | 单星放大框 | 见下方 | — |

### 单星内部（核心放大）
```
任务到达(Poisson λ) ─▶ [前向队列 Q^F_n] ─▶ ◇卸载决策◇
                                          ├─0─▶ [计算队列 Q^B_n] ─▶ [DVFS CPU f_cmp] ─▶ 完成
                                          └─k─▶ [ISL→邻居 m_k 的 Q^F]
```
- 任务 `K_i={D_i bits, X_i cyc/bit, deadline}`
- 动作 `a∈{0=本地, 1..4=转发邻居}`
- DVFS `f_cmp=min(F_max, max(√(Q^B/3V_DVFS κ), 截止下限))`
- 电池 `solar 充入 → DoD D_n∈[D_min,D_max] → 计算/传输能耗放出`，标 `health loss H_n`

### 箭头（3 种流，不同颜色/线型）
| 流 | 颜色 | 路径 |
|---|---|---|
| 任务流 | 黑实线 | 到达→Q^F→决策→(本地 Q^B→CPU→完成 / 转发→邻居) |
| 能量流 | 橙虚线 | 太阳→电池→(计算+传输能耗)，电池→DoD/HL |
| 信息广播 | 灰点线 | 各星状态→4 邻居（get_info）|

---

## 图 2：LyaMAPPO 框架图（Method — Lyapunov ⊗ MAPPO, CTDE）

### 整体布局（左→右数据流）
环境 → Lyapunov 奖励构造 → MAPPO 学习器(Actor/Critic) → PPO 更新（回流箭头）

### 模块清单
| 区 | 模块 | 内容/标注 |
|---|---|---|
| A 环境 | 卫星 + 三队列 | `Q^F_n, Q^B_n` + **`z_n`(DoD 虚拟队列, Neely)** ⭐2 |
| B Lyapunov 层 | Lyapunov 函数 | `L=½Σ_n[(Q^F)²+(Q^B)²+η z_n²]`, **η=0.5** |
| | drift-plus-penalty | `ΔL + V·Cost`, **V=50** → 每槽代价 → 奖励 `r_n` ⭐1 |
| | outcome-aware 增强 | `R_n=r_n + w_d·完成 − w_t·超时 − w_r·拒绝 − w_h·HL − w_q·队列`，权重 `10/5/5/2/0.05` ⭐3 |
| C Actor(分布式执行) | 输入 **54 维** per-task | `[ID(1)+本地(10,含 z_n)+邻居(9×4)+任务(7)]` |
| | 网络 | `LayerNorm→256→256→5→MaskedSoftmax→采样` |
| | 顺序决策 | 每 task 现拉最新状态(act_one) ⭐5 |
| D Critic(中心化训练) | 输入 **245 维** | `[5 节点×47 + 全局摘要10]`，**与 N 解耦** ⭐4 |
| | 网络 | `LayerNorm→256→256→1` |
| E PPO 更新 | GAE + 任务级优势分解 | `A_task=A_slot+β·(r−r̄)/σ`, **β=0.5** ⭐7 |
| | clipped PPO | `γ=0.99, λ_GAE=0.95, ε=0.2, β_e=0.02` |

### CTDE 边界（竖虚线分隔）
- 左/执行侧：每星本地跑 Actor（只需局部 54 维）→ 分布式
- 右/训练侧：Critic 用全局信息（245 维）→ 仅训练时

### 关键箭头
```
环境状态 ─▶ Actor(54) ─▶ 动作 a ─▶ 环境演进(DVFS/电池/队列)
                                   │
     ┌──── 奖励 R_n (Lyapunov+outcome) ◀──┘
     ▼
Rollout Buffer ─▶ GAE/任务级优势 ─▶ PPO ─▶ 更新 Actor & Critic（回流虚线）
```

### 7 创新点（图角 callout，编号对应 ⭐）
1. Lyapunov-MAPPO 深度耦合 2. z_n 入 Actor 状态 3. outcome-aware 奖励
4. 可扩展 Critic(与 N 解耦) 5. 顺序决策 6. 共享 DVFS 物理基底 7. 任务级优势分解

---

## 画图建议
- 工具：draw.io（矢量、免费）或 TikZ（投稿质量最高）。
- 配色：环境=灰蓝、Lyapunov=橙、Actor=红、Critic=绿（与 figures 里 LyaMAPPO 红呼应）。
- 创新点用 ⭐+编号贴对应模块，正文 METHODS 4.13 一一对应。
- 图 1 强调物理（DVFS/电池/ISL），图 2 强调算法（Lyapunov→奖励→MAPPO→更新）。
