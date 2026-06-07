# 技术路线图：MAPPO 在卫星 MEC 任务卸载中战胜 MHSPO

> 论文场景：LEO 25 颗 + λ_high=4 高过载
> 论文目标：CR ≥ MHSPO，HL 比 MHSPO 低 ≥ 30%，DoD 落在 [0.20, 0.40]

## 一、整体演化路线

```
┌─ M0: 起始（commit 7198ac6）──────────────────────────────────────┐
│ - CPU 模型：Zhang 公平均分                                        │
│ - H ∈ [100, 300]                                                  │
│ - 无 DVFS                                                         │
│ - 已发现 DoD=0.64-0.78 偏高问题                                   │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M1: DVFS 物理模型重构（eae53f6, 7198ac6）────────────────────────┐
│ - 引入 Li-style 整星 DVFS：f_cmp = sqrt(Q/(3·V·κ)) ∨ floor       │
│ - H 改为 Li 表 [10, 30]                                          │
│ - 新增 dvfs.py 模块                                              │
│ - 结果：DoD 落入 [0.30, 0.37] ✓                                   │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M2: Cost 函数 DVFS-aware 化（43c84f2, 001e48d）──────────────────┐
│ - delta_dod_comp 改为 DVFS 前后能耗差分                          │
│ - lyapunov.py queue_item 修复 THETA_NUM 偏置                      │
│ - 暴露 last_cpu_freq 到诊断 info                                  │
│ - 结果：LyapunovGreedy 仍卡 70% 平台（与决策无关）                 │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M3: Codex Stages 1-4（773c159）─────────────────────────────────┐
│ 1. eval_timeout 累加 bug 修复                                     │
│ 2. Sequential decision：每 task 现场拉 state                      │
│ 3. Outcome-aware reward：done/timeout/reject/hl/queue 5 项        │
│ 4. Actor state 54 维（含 DVFS、solar、slack_ratio）               │
│ - 结果：MAPPO CR=51%, HL 2.71e-4（HL 已达标）                     │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M4: Codex Stages 5+6（d2c1a6d）─────────────────────────────────┐
│ 5. Critic 加 10 维全局摘要（235→245，星座规模无关）              │
│ 6. Per-task advantage：A_task = A_slot + 0.5·(r_t − mean)/std     │
│ - 结果：CR=52%, HL 2.54e-4（与 M3 几乎打平）                      │
│ - 结论：Stage 5+6 不是当前瓶颈                                    │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M5: Reward Ledger 诊断（9664f17, fa5fccb）──────────────────────┐
│ - env.py 加 reward_ledger 字典记录各组分                          │
│ - recorder.py 聚合到 eval_runs.json                               │
│ - 发现：W_HL=10 让 HL penalty=−537 vs done=+49（19× 压制）        │
│ - 结论：reward 权重是真正瓶颈                                     │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M6: P0 Sweep & 定稿配置（本次）─────────────────────────────────┐
│ - Sweep BETA ∈ {0.15, 0.02} × W_HL ∈ {10, 5, 2}                  │
│ - 胜出配置：BETA=0.02, W_DONE=10, W_HL=2, W_TIMEOUT=5             │
│ - 8K 步即达到 CR=78.5%，HL −51% vs MHSPO                          │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌─ M7: 定稿 32K 训练（进行中）──────────────────────────────────────┐
│ - 32K 训练 + 5 baseline 完整对比                                  │
│ - 输出 5 指标曲线图                                              │
│ - 预期：CR ≥ 80%（超 MHSPO），HL ≤ 3e-4（−40%+），DoD ~0.5      │
└─────────────────────────────────────────────────────────────────┘
```

## 二、关键技术决策与依据

### 决策 1：CPU 物理模型选 Li 整星 DVFS 而非 Zhang 公平均分

**问题**：原 Zhang 均分模型下 nb=1 时 P=κf³=80 W（远超净充电 14.5 W），DoD 必然爆炸。

**选项**：
- A) 截止时间驱动 DVFS（机械规则，不需 Lyapunov）
- B) 共享 Lyapunov DVFS 控制器（闭式解，所有策略共用）⭐
- C) f_cmp 作为策略动作（最纯，但要重训所有 baseline）

**选 B 的理由**：
1. 与 Li TSC 2024 思想一致，论文可引用
2. 不改动作空间，baseline 无需重训
3. 5 策略共享 DVFS 保证公平对比

### 决策 2：H 参数选 Li 的 [10, 30] 而非 Zhang 的 [100, 300]

**原因**：H=[100,300] 下 deadline floor 持续把 f_cmp 顶到 f_max，电池被压榨。改 H=[10,30] 后 floor 自然降，DVFS 在舒适区。

### 决策 3：Sequential decision 路径（Codex Stage 2）

**问题**：原 env.step 一次性 get_actions(全部 task)，task #2 看到的是 task #1 入队前的状态。

**修复**：MAPPO 走 sequential 路径，每 task 决策前现场拉 state，看到最新 nb_hat / q_cycles_hat / f_floor_hat。

### 决策 4：Outcome reward 替代纯 Lyapunov cost（Codex Stage 3）

**问题**：原 reward = −Lyapunov_cost，MAPPO 只能间接通过 cost 学到 CR/HL。

**修复**：增加 5 项直接 outcome：
```
r_slot = Σ_task(−cost) + W_DONE·satisfied − W_TIMEOUT·timeout
       − W_REJECT·reject − W_HL·HL/HL_NORM − W_QUEUE·queue_pressure
```

### 决策 5：reward 权重定稿 W_DONE=10, W_HL=2（M6 sweep 决定）

**关键发现**：done/hl ratio 与 CR 强正相关
- W_HL=10 → done/hl=0.09 → CR=44.8%
- W_HL=5  → done/hl=0.45 → CR=64.7%
- W_HL=2  → done/hl=1.40 → CR=78.5% ⭐

### 决策 6：BETA=0.02（PPO entropy 系数）

**问题**：原 BETA=0.15 远超 PPO 标准（0.001-0.02），策略长期高随机性，QuickEval 38-63% 剧烈震荡。

**修复**：BETA=0.02，配合 reward 重平衡使用（单调 BETA 反而变差，证明 BETA 和 reward 是耦合问题）。

## 三、代码模块改动汇总

| 模块 | 改动 | 提交 |
|---|---|---|
| `core/dvfs.py`（新建）| Li-style Lyapunov 闭式选频 + deadline floor | eae53f6 |
| `core/satellite.py` | 54 维 Actor state + 47 维 critic per-node + DVFS state 暴露 | 773c159 |
| `core/env.py` | Sequential decision、outcome reward、reward ledger | 773c159, 9664f17 |
| `core/lyapunov.py` | DVFS-aware delta_dod_comp + queue_item 偏置修复 | 43c84f2, 001e48d |
| `core/constellation.py` | 全局摘要 get_global_summary() | d2c1a6d |
| `core/config.py` | DVFS 参数 + outcome weights + V_DVFS 校准 | 多次 |
| `training/networks.py` | Per-task advantage 计算 | d2c1a6d |
| `training/policy.py` | act_one + record_task_transition | 773c159 |
| `evaluation/recorder.py` | 5 项指标聚合 + reward ledger | 9664f17 |
| `train_mappo_lambda4.py` | 主训练入口 + CLI sweep 支持 | 9664f17 |

## 四、未做的改动 + 为何不做

| 改动 | 状态 | 不做的理由 |
|---|---|---|
| Stage 6 per-task advantage 提高 β | 已做 | β=0.5 已设，未证实 β 更大有用 |
| NO-OP 动作（延迟决策）| 未做 | M6 reward 修复已突破，可不必要 |
| LSTM/GRU Actor | 未做 | 暂未需要——若 M7 仍不达标会启用 |
| DOGD 注入 MAPPO state | 未做 | 与 MHSPO 区分度弱 |
| 全 critic GNN | 未做 | flat MLP + global summary 已经够 |
| 长训 64K-128K | 未做 | M6 8K 已 78.5%，32K 应该够 |

## 五、定稿配置（M7 用）

```python
# Config 关键超参
KAPPA       = 1e-26              # Zhang 原值
CPU_FREQ    = 2e9                # F_CMP_MAX
H_MIN/MAX   = 10 / 30            # Li 表
LAMBDA_HIGH = 4.0                # 主场景

# DVFS（V_DVFS 自动校准）
V_DVFS      = Q_max/(3·F_MAX²·κ) ≈ 7.5e16

# MAPPO 网络
state_dim         = 54            # Actor input
critic_state_dim  = 245           # 47×5 + 10 全局摘要
hidden_dim        = 256
HIDDEN layers     = 2 + LayerNorm

# PPO
GAMMA       = 0.99
LAMBDA_GAE  = 0.95
EPSILON     = 0.2
BETA        = 0.02               ⭐ 从 0.15 改
LR_ACTOR    = 1e-4
LR_CRITIC   = 1e-3
K_ROLLOUT   = 64
MINIBATCH   = 64
EPOCH       = 2

# Outcome reward 权重
W_DONE      = 10                 ⭐ 从 5 改
W_TIMEOUT   = 5                  ⭐ 从 3 改
W_REJECT    = 5                  ⭐ 从 3 改
W_HL        = 2                  ⭐ 从 10 改
W_QUEUE     = 0.05
HL_NORM     = 1e-4
```

## 六、5 项标准评估指标（论文用）

| 指标 | 定义 | 单位 | 论文期望 |
|---|---|---|---|
| **CR** | 完成数 / 到达数 | – | MAPPO ≥ MHSPO |
| **满意度** | 完成且 ≤ ddl / 完成总数 | – | MAPPO ≥ MHSPO |
| **时延** | (finish_slot − arrive_slot) × τ | s | 越低越好 |
| **HL/slot** | health_loss 累积 / 时隙数 | – | MAPPO ≤ 0.7·MHSPO |
| **平均 DoD** | 全星座平均放电深度 | – | [0.20, 0.40] |
| **队列积压** | (avg_qf + avg_qb) | MB | 越低越好 |
