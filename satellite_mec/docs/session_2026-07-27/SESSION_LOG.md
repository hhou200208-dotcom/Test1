# 会话记录 — LyaMAPPO/BLA-MAPPO 项目（2026-07-27）

> 主场景：**λ_high=4.0（重过载）, 25 卫星（5×5 子星座）, T_EVAL=5400 时隙, N=192 投影**。
> 主指标：**用户满意度 = satisfied/(done+timeout)**；外延量（能耗/时延/队列）= N=25 实测 ×(192/25)。
> 分支：`claude/upbeat-volta-6bk0y`。

---

## 0. 本会话 TL;DR

1. **诚实核对了 LyDRL-DoD（Zhong IoT-J 2026 的 MADDPG 卸载学习器）基线**，坐实其在过载下**双稳态**（0.83 高吞吐 / 0.41 低吞吐），并证明"过载下平均 DoD 由工作量钉死、跨吞吐不可比"。
2. **忠实复现了 Zhong eq41 reward**（`−Q·l −Y·(T−Tmax) −υ·D`，队列漂移驱动吞吐、无 W_DONE）；υ-sweep 仍双稳态。
3. **新写并训练了 MADRL-DoD**（MAPPO 学习器 + 「满意度 + 平均DoD」目标），W_DOD sweep {8,20,50} **全部落高吞吐/高 HL 簇，未达目标**——证明 HL 命门来自"择时（z_n/L'(δ)）"，不来自 DoD/学习器。
4. **主方法更名 LyaMAPPO → BLA-MAPPO**（display 名，未动金牌 checkpoint）。
5. 产出 **训练收敛曲线**（真实记录的奖励，conv3 run）与 **BLA-MAPPO 16K-vs-32K 训练预算消融图**。
6. 产出 **n=10 多种子「系统队列积压随时间」对比图**（本文件末尾）。

**核心结论（可入论文）**：BLA-MAPPO 是电池健康（HL）维度的断层赢家（HL≈1.77e-4，强策略中唯一 <2e-4），靠 z_n 放电择时；这一优势**与操作点无关、DoD-only 方法结构性够不着**。

---

## 1. 主对比结果（λ=4, T=5400, N=192）

n=10（mean±std，来源 `docs/multiseed_lh4.json` + `EXPERIMENT_CHAPTER.md` C.1）；LyDRL-DoD 为 n=1/n=5，单独标注。

| 方法 | 满意度 | HL/slot | 系统 cumHL | 系统能耗(kJ) | 系统队列(tasks) | E2E时延(s) | DoD |
|---|---|---|---|---|---|---|---|
| **BLA-MAPPO**(=LyaMAPPO 金牌) | **0.787±0.001** | **1.77e-4** | **183±4** | **6957±61** | 545±3 | 3.33 | 0.488 |
| LyDRL-DoD（MADDPG, Zhong）†n=1 | 0.834 | 5.17e-4 | ~537 | ~13072 | ~467 | 2.99 | 0.480 |
| MHSPO | 0.798±0.001 | 5.10e-4 | 529±5 | 12907±88 | 540±3 | 3.53 | 0.461 |
| GDCO | 0.727±0.001 | 2.50e-4 | 259±5 | 7915±62 | 692±3 | 4.52 | 0.475 |
| LSO（LocalOnly） | 0.269±0.004 | 1.86e-4 | 193±4 | 4852±72 | 1064±6 | 4.52 | 0.371 |
| （TD3-Sched，历史） | 0.838±0.001 | 4.47e-4 | 463±6 | 12217±96 | 482±3 | 3.06 | 0.484 |
| （MAPPO-NoDOD，消融） | 0.824±0.001 | 5.34e-4 | 554±6 | 13238±92 | — | — | 0.477 |

**读表**：无任一策略支配 BLA-MAPPO——TD3/MHSPO 赢吞吐 1–5pp 却在 HL/能耗上 2.5–3×；BLA-MAPPO 让 ~5pp 吞吐换 **HL≈1/3、能耗≈一半**。在不可逆的电池健康维度断层第一。

### 反直觉铁证（论文金句）
- **BLA-MAPPO 平均 DoD 最高（0.488）却 HL 最低（1.77e-4）** → 低 HL ≠ 低 DoD，靠 **z_n 放电择时**（把耗能推到低 δ 时刻）。
- LSO 能耗最低但 HL≈BLA-MAPPO（在 δ=0.8 高位硬放）→ **少花能量 ≠ 少伤电池**。

---

## 2. 本会话过程（按时间）

### 2.1 LyDRL-DoD 基线的诚实呈现
- git 真实状态：committed 的是 **0.83 高吞吐版**（`MADDPG_DoD_lh4_32K`, max-CR, sat 0.834/HL 5.17e-4）；"0.41 定稿版"从未提交到本分支。
- 实测**双稳态**：0.83（DoD 0.486，不降 DoD）/ 0.41（DoD 0.393，低 DoD 是低吞吐副产品）；中间态（0.75）任何奖励旋钮够不着。
- 关键洞察：**DoD/HL/能耗随工作量缩放，跨吞吐不可比**；有效对比要匹配吞吐或用强度指标（HL/slot、HL/完成任务）。
- **HL/完成任务**：LyDRL-DoD 两吸引子都 ~2.8–3.1× BLA-MAPPO → **命门与操作点无关**。

### 2.2 为什么"没有中间值"（双稳态机理）
过载下"吞吐 vs 省电"是带**拥塞崩溃正反馈**的双稳态（fold 分岔）：队列涨→超时级联→有效吞吐降→队列更涨；f³ DVFS 非线性把边界磨成跳变。中间态动力学不稳定，故不可达。用户要的"中吞吐+低伤电"不在这条轴上，而在 z_n 的"择时"轴上。

### 2.3 为什么"优化 DoD 却 HL 垃圾"（实现诊断）
- 旧 MADDPG-DoD 的 DoD 惩罚 `ETA·(z/Z_MAX)·(ΔDoD/ΔDoD_MAX)` 被归一化压到 **≈完成奖励的 0.1%**（实测 avg_z/Z_MAX≈0.036）→ 形同虚设，**未忠实 Zhong**。
- **忠实复现 Zhong eq41**（`baselines/maddpg_dod.py` 加 `zhong_reward` 模式 + Y_n 时延虚拟队列 + `--zhong/--upsilon`）→ υ-sweep 仍双稳态（撞 GDCO 的 DoD 地板 0.475）。
- 结论：**过载下平均 DoD 由工作量钉死，任何 DoD-only 方法都到 ~0.475 地板，不是 bug**。

### 2.4 MADRL-DoD（新写，MAPPO + 满意度+平均DoD 目标）
- `train_mappo_lambda4.py --madrl_dod --w_dod <W>`；奖励 = `W_DONE·满意度 − W_DOD·δ`，无 HL/L'(δ)。
- **W_DOD sweep {8,20,50}（8K each）结果（未达目标）**：

| W_DOD | 满意度 | HL/slot | DoD |
|---|---|---|---|
| 8 | 0.827 | 5.40e-4 | 0.492 |
| 20 | 0.822 | 5.29e-4 | 0.480 |
| 50 | 0.838 | 5.17e-4 | 0.480 |
| 目标 | ~0.75 | ~2.42e-4(GDCO) | — |

- 6× W_DOD 只降 HL ~4%，全卡高吞吐/高 HL。δ-存量罚信用分配太弱、学不出负载均衡。**证明 HL 命门 = z_n 择时目标，不是学习器、不是 DoD 罚。**

### 2.5 更名 + 训练预算消融
- LyaMAPPO → **BLA-MAPPO**（display 名，金牌 checkpoint 未动）。
- **16K vs 32K 训练预算消融**（同方法不同步数，`LyaDRL_DoD` = full-LyaMAPPO@16K，sat 0.715/HL 2.13e-4）→ 图见 `docs/figures_ablation_16k/`。**注：这是同方法@16K,不是独立基线。**

### 2.6 收敛曲线 + 队列 n=10 图
- 训练奖励收敛曲线：`evaluation/runner.py` 加每-槽奖励记录（`reward_curve.json`）；BLA-MAPPO 约 **100 episode 内快速收敛**（样本高效：密集奖励 + 小离散动作 + 每槽大量任务样本 + PPO on-policy）。
- n=10 多种子「系统队列积压随时间」对比图（`docs/figures_queue_n10/queue_backlog_n10.png`，数据 `docs/queue_n10_data.json`）。系统队列稳态(tasks, n=10)：**LyDRL-DoD 467 < MHSPO 540 ≈ BLA-MAPPO 545 < GDCO 692 ≪ LSO 1069**。BLA-MAPPO 队列有界稳定（Lyapunov drift 保 mean-rate stability），与 C.1 表 n=10 数一致。

---

## 3. 代码清单（本会话新增/改动）

| 文件 | 说明 | 状态 |
|---|---|---|
| `baselines/maddpg_dod.py` | 加 `zhong_reward` 忠实 eq41 模式 + Y_n 时延队列 + `--w_done/--eta` 生效 | 改 |
| `train_maddpg_dod.py` | 加 `--w_done/--eta/--zhong/--upsilon/--t_max` | 改 |
| `train_mappo_lambda4.py` | 加 `--dod_only`(LyDRL-DoD/MAPPO)、`--madrl_dod/--w_dod`(MADRL-DoD)、`--run_name` | 改 |
| `core/config.py` | 加 `UPSILON/T_MAX_DELAY/ZHONG_L_NORM/W_DOD` | 改 |
| `core/env.py` | 加 per-sat queue/energy/delay 到 info（Zhong reward 用）+ `−W_DOD·δ` 项（默认0） | 改（加法式，不影响 LyaMAPPO） |
| `evaluation/runner.py` | 训练时记录每-槽奖励 → `reward_curve.json` | 改（加法式） |
| `focused_eval_maddpg.py` | checkpoint 路径参数化 | 改 |
| `plot_ablation_16k32k.py` | 16K-vs-32K 消融图（评 16K ckpt + 出 5 图） | 新 |
| `plot_reward_convergence.py` | 奖励收敛曲线（读 reward_curve.json） | 新 |
| `eval_queue_n10.py` | n=10 多种子队列时序评估 + 出图（断点续跑） | 新 |

## 4. 数据清单

| 文件 | 内容 |
|---|---|
| `docs/reward_curve_bla_mappo.json` | BLA-MAPPO 32K 训练每-槽奖励（32000 值）；每-episode = reshape(500,64).sum(1) |
| `docs/series_bla16k.json` | BLA-MAPPO@16K 评估 per-slot series（16K 消融图用） |
| `docs/queue_n10_data.json` | n=10 多种子每-槽队列均值/std（5 方法） |
| `docs/multiseed_lh4.json`（已存） | n=10 标量（满意度/HL/能耗/队列 mean±std）；能耗为 N=25,×192/25 得 N=192 |
| `docs/scoreboard7_lh4.json`（已存） | n=1 标量（含 MADDPG_DoD/LyDRL-DoD） |

## 5. 图清单

| 目录 | 图 |
|---|---|
| `docs/figures_ablation_16k/` | 16K-vs-32K 消融 5 图（satisfaction/energy/delay/cumHL/queue） |
| `docs/figures_convergence/` | 奖励收敛曲线（reward_convergence*.png） |
| `docs/figures_queue_n10/` | n=10 系统队列积压随时间对比图 |

---

## 6. 诚实备注 / 待办（验收关注）

1. **收敛曲线的 run 一致性**：奖励收敛曲线来自 conv3 run，其最终 **CR=0.64 < 金牌 0.786**（种子波动）。收敛形状有效，但若论文报 0.786，应让"曲线来源 run = 报告 run"一致（做一次定稿 32K：eval 数进表、其曲线进图）。
2. **奖励为正值**（Zhong 为负）：奖励定义不同（我们含 W_DONE 完成激励），形状（升+plateau）一致,勿强改符号。
3. **2000 episode 不会更好**：~ep100 已收敛,加长只是"图更密",非性能提升。真优化杠杆：多种子选优、试 **V=100**（项目内 V 扫描显示 CR+1.3pp、HL−0.11e-4）。
4. **命名边界（诚信）**：`LyaDRL_DoD`(16K) = BLA-MAPPO 同方法欠训版,**不可当独立基线与 BLA-MAPPO 并列比较**（自己比自己）。仅作"训练预算消融",且需在论文注明。
5. **LyDRL-DoD 主对比数为 n=1/n=5**；若进正式表建议补 n=10。

## 7. 复现关键命令
```bash
cd satellite_mec
# 忠实 Zhong eq41 baseline
python train_maddpg_dod.py --zhong --upsilon <υ> --t_train 8000 --tag zh
# MADRL-DoD（MAPPO + 满意度+DoD）
python train_mappo_lambda4.py --madrl_dod --w_dod <W> --w_done 10 --beta 0.02 --t_train 8000 --skip_baselines --tag madrl
# 奖励收敛曲线
python plot_reward_convergence.py <results/.../reward_curve.json>
# n=10 队列时序图
python eval_queue_n10.py
```
