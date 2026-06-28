# 算法整体流程（Overall Algorithm Flow）— 方法章草稿 · 中文 · CCF-A 版式

> 本节原属实验草稿 §4，按 CCF-A 惯例移入**方法章**（实验部分仅含设置 + 结果分析）。
> 对应框架图 `docs/figures_paper/arch_lyamappo_framework.png`（投稿质量 TikZ 源 `docs/tikz/arch_lyamappo_framework.tex`）。
> 超参引用见实验草稿 §1.6（或方法章对应超参表）；数值与 `ARCHITECTURE_DIAGRAMS_SPEC.md` / 代码核实一致。
> 编号占位为 `X`，并入正式方法章时替换为实际章节号。

---

## X 算法整体流程（Overall Algorithm Flow）

本节给出 LyaMAPPO 的整体流程，对应框架图 `arch_lyamappo_framework.png`。算法采用**集中训练、分布式执行（CTDE）**范式，将 Lyapunov drift-plus-penalty 与 MAPPO 深度耦合。

### X.1 框架总览

数据流自左向右分四个阶段（图中以竖虚线标出 CTDE 边界）：

\[
\text{环境} \;\rightarrow\; \text{Lyapunov 奖励构造} \;\rightarrow\; \text{MAPPO 学习器(Actor/Critic)} \;\xrightarrow{\text{PPO 回流}}\; \text{参数更新}
\]

- **环境（A 区）**：每颗卫星维护三个队列——前向队列 \(Q^F_n\)、计算队列 \(Q^B_n\)，以及**放电深度虚拟队列 \(z_n\)**（按 Neely 的虚拟队列法将 DoD 约束转化为队列稳定性）。
- **Lyapunov 层（B 区）**：构造 Lyapunov 函数 \(L = \tfrac12\sum_n\big[(Q^F_n)^2 + (Q^B_n)^2 + \eta\, z_n^2\big]\)（\(\eta=0.5\)），最小化 drift-plus-penalty \(\Delta L + V\cdot \text{Cost}\)（\(V=50\)）得到每时隙基础代价/奖励 \(r_n\)；再叠加 **outcome-aware 奖励塑形** \(R_n = r_n + W_{\text{done}}\!\cdot\!\text{完成} - W_{\text{to}}\!\cdot\!\text{超时} - W_{\text{rej}}\!\cdot\!\text{拒绝} - W_{\text{HL}}\!\cdot\!\text{HL} - W_{\text{queue}}\!\cdot\!\text{队列}\)（权重见超参表）。
- **Actor（C 区，分布式执行）**：输入 54 维（含本地 \(z_n\)），网络 `LayerNorm→256→256→5→MaskedSoftmax`，逐任务采样动作 \(a\in\{0=\text{本地},\,1\text{–}4=\text{转发邻居}\}\)。
- **Critic（D 区，集中训练）**：输入 245 维（5 节点局部 + 10 维全局摘要，**与 N 解耦**），网络 `LayerNorm→256→256→1`，仅训练时使用全局信息。
- **PPO 更新（E 区）**：GAE + 任务级优势分解，clipped PPO 回流更新 Actor 与 Critic。

### X.2 决策流程（分布式执行）

执行侧每星只需本地 54 维状态、独立运行 Actor，无需中心节点——故可扩展到 N=192。每时隙、每任务**顺序决策**（关键创新：每个任务现拉最新局部状态 `act_one`，使同一时隙内先后任务能感知彼此造成的队列/电池变化）：

```
对每个时隙 t:
  对前向队列 Q^F_n 中每个任务 i（按序）:
    s_i ← 拉取最新局部状态(54 维, 含 z_n, 邻居 9×4, 任务 7)
    logits ← Actor(s_i);  动作掩码屏蔽不可行邻居
    a_i ← MaskedSoftmax(logits) 采样
    若 a_i = 0: 任务入本地计算队列 Q^B_n
    否则:        任务经 ISL 转发至邻居 a_i 的 Q^F
  环境演进: DVFS 闭式解定频 f_cmp → 计算/传输能耗 → 电池 DoD/HL 更新 → 队列推进
```

### X.3 训练流程（集中训练）

训练侧用全局信息的 Critic 估值，按 PPO 更新（\(\gamma=0.99\)、\(\lambda_{\text{GAE}}=0.95\)、\(\epsilon=0.2\)、\(\beta_e=0.02\)、\(\beta_{\text{task}}=0.5\)、\(K=64\)、minibatch=64、epoch=2）：

```
初始化 Actor θ, Critic φ
重复直到 32K 时隙:
  # 采样
  以当前 θ 跑 K=64 时隙, 记录 (s_i, a_i, R_n, 全局状态) 入 Rollout Buffer
  # 优势估计
  V(·) ← Critic_φ(全局状态)
  A_slot ← GAE(R_n, V; γ=0.99, λ=0.95)
  A_task ← A_slot + β_task·(r − r̄)/σ        # 任务级优势分解, β_task=0.5
  # 更新 (epoch=2, minibatch=64)
  L_actor  ← clipped-PPO(θ; A_task, ε=0.2) − β_e·H(π)     # 熵正则
  L_critic ← MSE(Critic_φ, 回报目标)
  θ ← θ − lr_a·∇L_actor   (lr_a=1e-4)
  φ ← φ − lr_c·∇L_critic  (lr_c=1e-3)
```

### X.4 七个创新点

框架图以 ⭐ 标注七个创新点，与上述流程一一对应：① Lyapunov–MAPPO 深度耦合（drift-plus-penalty 直接构造奖励）；② \(z_n\)（DoD 虚拟队列）入 Actor 状态；③ outcome-aware 奖励塑形；④ 可扩展 Critic（与 N 解耦，使 N=25 训练能投影至 N=192）；⑤ 任务级顺序决策（`act_one`）；⑥ 共享 DVFS 物理基底（保证对比公平）；⑦ 任务级优势分解。其中 ②④⑤ 是实验 §2 电池优势（择时）与可扩展性的算法根源，⑥ 是全部对比/消融公平性的前提。
