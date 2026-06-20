# 新会话开场提示

> 用法：开新对话后，把下方虚线之间的全部内容复制到第一条消息。

---

你好。我们继续 **LyaMAPPO 论文项目**。

## 仓库

- 远程：`hhou200208-dotcom/Test1`
- 分支：`claude/upbeat-volta-6bk0y`
- 工作目录：`/home/user/Test1/satellite_mec/`

## 必须立刻读的 4 份文档

```
satellite_mec/WORK_LOG.md          # M0-M11 完整工作历史 + 用户需求
satellite_mec/PROJECT_STATE.md     # 项目状态快照
satellite_mec/METHODS.md           # 论文方法章节 4.1-4.13
satellite_mec/checkpoints/README.md # 模型加载指南
```

读完即秒回状态，**不要从零问我**。

## 沟通规范（爸爸要求）

- 称呼我：**爸爸**
- 风格：**直接 + 量化 + 表格化**，不要废话
- 训练时**每 5 分钟主动汇报进度**
- 绝不：**编造引用 / 假装跑过实验 / 隐瞒失败**

## 本次会话的目标

**补充 baseline**，要求：

1. **2024-2026 年发表，CCF-A 优先**
2. 与 LyaMAPPO 场景兼容（LEO 25 颗卫星，λ_high=4.0，task offloading，DVFS 物理）

### 上次会话已经推荐的两篇候选

```
Chen, Yang, Hu, Wu, Huang. "A Game-Theoretical Approach for
Distributed Computation Offloading in LEO Satellite-Terrestrial
Edge Computing Systems." IEEE Transactions on Mobile Computing,
vol. 24, no. 1, pp. 363–378, Jan. 2025.
DOI: 10.1109/TMC.2025.3526200

Xie, Cui, Ho, He, Guizani. "Computation Offloading and Resource
Allocation in LEO Satellite-Terrestrial Integrated Networks With
System State Delay." IEEE Transactions on Mobile Computing, 2025.
```

也可以选择重新搜索更合适的。

## 不能动的"红线"

| 项 | 状态 | 说明 |
|---|---|---|
| LyaMAPPO V=50 32K checkpoint | ✅ 已锁定（论文定稿）| `checkpoints/LyaMAPPO_lh4_32K/` |
| METHODS.md 章节 4.1-4.13 | ✅ 论文方法章节，**不改** | |
| 7 项核心创新（见 METHODS.md 4.13）| ✅ **绝对不能丢** | |
| 已有 5 baseline 实现 | ✅ 不动 | `baselines/` 目录 |
| BETA=0.02, W_DONE=10, W_HL=2, ... | ✅ M6 sweep 定稿，**不改** | |

## 当前 LyaMAPPO 论文 scoreboard（vs MHSPO）

```
CR     0.787 vs 0.798   −1.1 pp (统计显著, n_runs=3 验证)
Sat    0.789 vs 0.802   −1.3 pp
Delay  3.33 s vs 3.54 s ✅ −0.21 s
HL     1.77e-4 vs 5.09e-4 ✅ −65 %
DoD    0.496 vs 0.469
Queue  93.7 MB vs 90.7 MB ≈ 平
```

## 加 baseline 的实现路径（建议）

参考 `baselines/mhspo.py` 的结构（也是基于论文实现的）：

1. 继承 `PolicyInterface`
2. 实现 `get_actions(obs, masks) -> Dict[sat_id, List[action]]`
3. 算法逻辑放在 policy 类内
4. 放到 `baselines/<new_baseline>.py`
5. 在 `baselines/__init__.py` 暴露

集成进 `train_mappo_lambda4.py` / `eval_multi_runs.py` 的 5 策略列表即可。

## 第一步：你要做的

1. 读完上述 4 份文档
2. 用 `git log --oneline -15` 看最近 commit
3. 列出新 baseline 候选的**简要 abstract + 实现复杂度评估**
4. 让爸爸选实现哪个

不要直接开始改代码，先汇报理解。

---

**结束提示**。开始干活前先确认你已经完成上述阅读。
