# 实验（中文草稿 · CCF-A 版式）

> 结构按 `experiment_outline.md`：§2 对比实验（5 算法 × 5 指标）→ §3 消融实验（2 变体 × 4 指标）→ §4 算法流程图。§1 实验设置为独立成章所必需的前置。
> 数值口径：所有标量取自 `docs/multiseed_lh4.json`（n=10 独立种子，报 mean±std）与 `docs/diagnostics_lh4.json`（机理三因子分解，代表性单次诊断）。引用一律以 `\cite{...}` 占位。
> 诚实框架：Pareto 权衡（非无限定"综合最优"）；电池健康损失 HL 与能耗为两个独立维度；N=25 实测、N=192 为已验证线性可扩展性投影；训练为单 checkpoint，方差作为局限披露。
>
> **进度**：§1 ✅ 待审 ｜ §2 ⬜ ｜ §3 ⬜ ｜ §4 ⬜

---

## 1 实验设置（Experimental Setup）

### 1.1 仿真场景与星座

实验在低轨（LEO）Walker 星座的星地协同计算卸载场景下进行。论文目标规模为 **N=192**（16 个轨道面 × 12 颗/面）。受训练算力所限，**策略的训练与逐时隙评估在 N=25（5×5）子星座上完成**，系统级外延指标（累计健康损失、总能耗、总积压）按已验证的线性可扩展性投影到 N=192（投影口径见 §1.4，可扩展性验证见 §1.4 与图 `scalability.png`）。

为刻画空间负载不均衡与由此产生的电池压力，星座中 **1/5 的卫星为高负载星**（N=25 时为 5 颗，星 ID `[4, 6, 12, 22, 24]`），任务到达率 **λ_high = 4.0**（高过载）；其余 20 颗为低负载星，**λ_low = 0.1**。任务到达服从泊松过程。主战场设为 λ_high = 4.0——刻意制造热点，使卸载决策与电池管理成为必要而非可选。

轨道周期为 5400 时隙，时隙长度 τ = 1 s。每次评估时长 **T_eval = 5400 时隙**，在 **n = 10 个独立随机种子**上重复，报告 mean ± std。

### 1.2 任务、计算与链路参数

| 项 | 取值 | 来源 |
|---|---|---|
| 任务大小 \(D_i\) | 10–50 Mb（均匀分布） | — |
| 计算密度 \(X_i\) | 10–30 cycles/bit | \cite{li_tsc_2024} |
| 截止时间 | 1–12 s | — |
| 最大转发跳数 | 3 | — |
| CPU 频率上限 \(f_{\max}\) | 2 GHz | \cite{zhang_tmc_2023} |
| 计算能耗系数 \(\kappa\) | \(1\times10^{-26}\) | \cite{zhang_tmc_2023} |
| 单星并发上限 | 6 | — |
| 星间链路速率 \(B\) | 100–300 Mb/s | — |
| 发射功率 \(P_T\) | 0.1 W | — |
| 每星邻居数 | 4（同轨前后 + 邻轨左右） | — |

**共享 DVFS 物理基底（公平性保证）**：所有对比与消融策略运行在同一套 Lyapunov-DVFS 频率选择、电池与星间链路模型之上，差异仅来自卸载/调度策略本身。计算频率由闭式解给出

\[
f_{\text{cmp}} = \mathrm{clip}\!\Big(\max\big(\sqrt{Q^B/(3\,V_{\text{DVFS}}\,\kappa)},\, f_{\text{floor}}\big),\, 0,\, f_{\max}\Big),
\qquad
E_{\text{cmp}} = \kappa\, f_{\text{cmp}}^{3}\, \tau,
\]

其中 \(f_{\text{floor}}\) 为由截止时间导出的频率下限，保证排队任务不被饿死（\cite{li_tsc_2024} 式(8)）。

### 1.3 电池与健康损失模型

| 项 | 取值 |
|---|---|
| 电池容量 \(E_{\text{cap}}\) | 36 kJ（10 Wh） |
| 太阳能功率 \(P_{\text{solar}}\) | ≤ 30 W |
| 日照比 | 0.65（日照/地影交替） |
| 星务基线功耗 | 5 W |
| 放电深度 DoD \(\delta\) | \([0.1,\,0.8]\) |
| 健康损失函数 | \(H(\delta) = \delta\cdot 10^{\,a(\delta-1)}\)，\(a = 0.8\) |

每时隙、每星按下式更新放电深度并累计健康损失：

\[
\delta \leftarrow \mathrm{clip}\!\Big(\delta + \frac{E_{\text{comp}}+E_{\text{trans}}+E_{\text{house}}-E_{\text{solar}}}{E_{\text{cap}}},\ [0.1,0.8]\Big),
\]
\[
\mathrm{HL} = L'(\delta_{\text{before}})\cdot \frac{E_{\text{comp}}+E_{\text{trans}}}{E_{\text{cap}}},
\qquad
L'(\delta) = 10^{\,a(\delta-1)}\big(1 + a\ln 10\cdot\delta\big),
\]

其中 \(L'(\delta)\) 为边际健康损伤率。**关键物理性质**：HL 对 DoD 指数凸、计算能耗对频率三次方凸——这是后文机理分析的物理根基（同样的能量在更低 \(\delta\) 时刻造成的不可逆损伤指数级更小）。

### 1.4 评估指标与口径

按 outline，对比实验报告五个指标；各指标口径如下（系统级量的 N=192 投影约定一并给出）：

1. **用户满意度（主指标）** \(=\) satisfied\(/\)(done + timeout)。完成率 CR \(=\) done\(/\)arrived 仅作内部核对（实测 CR ≈ 满意度，差 ≤ 0.3pp），不上结果台面。满意度为强度量，**不随 N 投影**。
2. **电池健康损失 HL**：per-sat per-slot 值；**系统累计 HL = per-sat 均值 × 192**。
3. **系统总能耗**：N=25 系统实测总量 × (192/25)，图注标注 "projected from validated linear scaling"。
4. **端到端时延**：每任务 E2E 时延均值，强度量，**不投影**。
5. **任务积压**：**系统总任务个数 = per-sat 均值 × 192**（任务个数口径，非字节）。

> **DoD 不作结果图**：各策略稳态 DoD 落在 0.37–0.50（实测，\cite{saft_ves16} 标称空间级锂电工作区间），仅在设置中披露；因 LyaMAPPO 优化目标是 HL 而非 DoD（机理见 §2.2），DoD 不进结果对比。

**可扩展性验证**：同一 checkpoint 在真实 N=192 星座上跑通，per-sat 量与速率呈 N-不变，故系统级量按 ×(192/25) 或 ×192 投影成立（图 `scalability.png` 为验证图）。N=25 原始数据与 checkpoint 保留作为投影证据。

### 1.5 对比算法与消融变体

所有方法共享 §1.2 的 Lyapunov-DVFS 物理基底，保证对比公平。

**对比 baseline（4 个）**

- **TD3-Sched**\cite{huang_tmc_2024}：学习型，取双时间尺度调度器的小尺度 TD3 调度部分；原文的资源切片/AEF/self-attention 在本单服务环境无落脚点（标注 N/A）。其代价函数**不含电池健康/DoD 项**（遵循原文）。checkpoint：`TD3Sched_lh4_16K`。
- **MHSPO**\cite{mhspo_ref}：多跳启发式卸载 + Lyapunov 队列控制（\(V_{\text{lyapunov}}=10\)），最强经典 baseline；评估前经预热以稳定其内部估计。
- **GDCO**\cite{chen_tmc_2025}：博弈论分布式计算卸载，非学习，各星按博弈均衡决策；**代价不含电池健康项**。
- **LSO（Local-only，仅本地）**：平凡下界，所有任务本地计算、零卸载。

**消融变体（2 个）**

- **MAPPO-NoDoD（− 电池感知）**：在 LyaMAPPO 基础上移除电池相关项。**诚实披露**：`--no_battery` 同时移除了 (i) Lyapunov 动作代价中的电池/DoD 项 与 (ii) outcome 奖励中的 \(W_{\text{HL}}\)（置 0），故为**合并消融**（界定"电池感知整体"的作用），非单项消融。
- **LyapunovGreedy（− 学习）**：共享同一 Lyapunov 代价计算器，但用无学习的贪心逐任务最小化当前时隙代价（无长视野信用分配、无 critic、无 outcome 塑形）。

### 1.6 训练配置与超参（LyaMAPPO 定稿）

> 注：`config.py` 默认值 ≠ 定稿值；定稿经训练 CLI 覆盖，精确命令归档于 `checkpoints/README.md`。

\(V = 50\)，\(\eta = 0.5\)；奖励权重 \((W_{\text{done}}, W_{\text{to}}, W_{\text{rej}}, W_{\text{HL}}, W_{\text{queue}}) = (10, 5, 5, 2, 0.05)\)。PPO 超参：\(\gamma = 0.99\)，\(\lambda_{\text{GAE}} = 0.95\)，\(\epsilon = 0.2\)，熵系数 \(\beta_e = 0.02\)，任务级优势 \(\beta_{\text{task}} = 0.5\)，actor 学习率 \(1\times10^{-4}\)，critic 学习率 \(1\times10^{-3}\)，rollout 长度 \(K = 64\)，minibatch \(= 64\)，epoch \(= 2\)。

Actor 输入 54 维（per task），Critic 输入 245 维（5 节点局部 + 10 维全局摘要，**与 N 解耦**）。LyaMAPPO 训练 32K 时隙。

---

## 2 对比实验（Comparison）

> ⬜ 待写：引言（5 算法定位 + Pareto 总览）→ 2.1 寿命损耗 → 2.2 系统时延 → 2.3 用户满意度 → 2.4 系统能耗 → 2.5 任务积压。

## 3 消融实验（Ablation）

> ⬜ 待写：MAPPO-NoDoD / LyapunovGreedy × (寿命损耗 / 系统时延 / 用户满意度 / 系统能耗)。

## 4 算法流程图（Algorithm Flowchart）

> ⬜ 待写：LyaMAPPO 训练与决策流程图说明（配 `arch_lyamappo_framework.png`）。
