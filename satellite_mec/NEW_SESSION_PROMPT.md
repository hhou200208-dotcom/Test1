# 新会话开场提示

> 用法：开新对话后，把下方虚线之间的全部内容复制到第一条消息。

---

你好。我们继续 **LyaMAPPO 论文项目**。请喊我**爸爸**。

## 仓库
- 远程：`hhou200208-dotcom/Test1`
- 分支：`claude/upbeat-volta-6bk0y`（所有产物都在这，已 push）
- 工作目录：`/home/user/Test1/satellite_mec/`

## 必须立刻读的文档（读完秒回状态，不要从零问我）
```
satellite_mec/WORK_LOG.md                  # 完整工作历史 M0-M13 + 论文写作约定
satellite_mec/PROJECT_STATE.md             # 状态快照
satellite_mec/METHODS.md                   # 论文方法 4.1-4.13（不改）
satellite_mec/checkpoints/README.md        # 模型加载
satellite_mec/docs/ARCHITECTURE_DIAGRAMS_SPEC.md  # 架构图构图清单
satellite_mec/docs/EXPERIMENT_CHAPTER.md   # 实验章节正文草稿(A-F,含机理三因子分解)
satellite_mec/docs/figures_paper/          # 论文图(14结果图 + 2架构图 arch_*)
satellite_mec/docs/figures_clean/          # 9 张极简图(无标题/无单位)
```

## 沟通规范（铁律）
- 称呼：**爸爸**；风格：**直接 + 量化 + 表格化**，不废话
- 训练时**每 ~5 分钟/每个里程碑主动汇报**
- **绝不：编造引用 / 假装跑过实验 / 隐瞒失败**（投影 N=192 必须标注"projected from validated scaling"）

## 当前进度（截至本次会话结束）
**已完成**：
1. **2 个新 baseline**：GDCO（Chen TMC2025 博弈论，`baselines/gdco.py`，非学习）+ TD3-Sched（Huang TMC2024 小尺度调度器，`baselines/td3_sched.py`，学习型，ckpt `TD3Sched_lh4_16K`）。
2. **叙事定稿=电池寿命**：LyaMAPPO 赢整个电池维度（系统累计 HL ~180 vs MHSPO 527/TD3 460，约 1/3；系统能耗 912kJ@N25 最低）。TD3 在满意度/时延上更强→走 Pareto + 电池命门叙事。
3. **指标口径**：主指标=**满意度**（CR 仅内部）；DoD 不进结果图（只 setup 披露，Saft VES16 30-50% 背书）；累计 HL=**系统总和(×192)**；队列=**系统总任务个数(per-sat×192)**。结果图标题已去掉 λ=4（只留 N=192，更干净）。
4. **N=192 框架**：实测验证 N-不变性（同 ckpt 跑通 192 颗）；系统总量×(192/25)投影、率不变；`scalability.png` 是验证图。
5. **14 张图**（`docs/figures_paper/`）：
   - 对比(LyaMAPPO/TD3/MHSPO/GDCO/**LSO**)：pareto · bars_4metrics · cumulative_hl · queue_backlog · satisfaction(+_pdf) · total_energy(+energy_pdf) · total_delay(+delay_pdf) · scalability
   - 消融(LyaMAPPO/**MAPPO-NoDOD**/LyapunovGreedy)：ablation_satisfaction · ablation_cumulative_hl · ablation_energy
   - 原始数据：`docs/series_lh4_n25.json`(逐槽) + `docs/scoreboard7_lh4.json`(标量)
   - 出图脚本：`plot_paper_figures.py`(对比) · `plot_paper_satisfaction.py`(pareto+柱状) · `plot_pdf_figures.py`(PDF) · `plot_abl_figures.py`(消融)
   - 重跑评估：`eval_series.py`(逐槽,8策略) · `eval_all7_scalars.py`
6. **架构图 2 张**（M14）：
   - PNG（matplotlib，**已跑通**）：`docs/figures_paper/arch_system_model.png` · `arch_lyamappo_framework.png`
   - TikZ 投稿源（**pdflatex 编译通过**，含 `.pdf`）：`docs/tikz/arch_system_model.tex` · `arch_lyamappo_framework.tex`
   - 脚本 `plot_arch_figures.py`；参数全对代码核实（54/245 维 · V=50 · 定稿权重 · DVFS 闭式解）；图 2 贴 ★1–7 创新点

**命名**：LocalOnly→显示 **LSO**；MAPPO_NoBat→显示 **MAPPO-NoDOD**（数据/ckpt key 不变，只图标签映射）；GreedyDelay 已从图中删除。

## 本会话新增（M14 之后，全部已 push）
- **Codex 对抗式审查**（`docs/CODEX_REVIEW_BRIEF.md`）：机理/代码/方法论三层。已修 6 个代码问题（beta_task→`Config.BETA_TASK`、TD3 注释中性化、超参命令归档进 checkpoints/README、`eval_diagnostics.py` 口径修正等）。Codex 判定："综合最优"不成立、应写 **Pareto 权衡**。
- **多种子 n=10 评估**（`docs/multiseed_lh4.json`，`eval_multiseed.py`）：对比/消融表已带 mean±std（CI 极紧）。LyaMAPPO 满意度 0.787（强策略**第4**），但系统累计 HL **183**/能耗 **6957kJ** 强策略最低 → **Pareto + 电池命门坐实**。
- **机理三因子分解**（`docs/diagnostics_lh4.json`，`eval_diagnostics.py`）：`cumHL=meanL'×timing×energy`；`能耗=level×dispersion`。本质=**z_n 择时(~1.5×) × 省能(~1.85×)**；削峰为次要、DoD 平均水平**未降**（已写进 EXPERIMENT_CHAPTER C.2）。
- **V 敏感性 32K 全扫描**（`docs/sensitivity_v_32k.json`）：5 点 CR 极差仅 2.4pp < **训练-seed 方差 ~4pp** → **V 在噪声内不可分辨**；结论改"V-鲁棒性"非"V-最优"。
- ⚠️ **金 checkpoint 钉死**：`checkpoints/LyaMAPPO_lh4_32K`（0.786/1.74e-4）**重训复现不出来**（普通 V=50 重训仅 0.745/2.89e-4）。已加 `CHECKSUMS.md5` + README 警告 + 本地 tag `golden-lyamappo-v50-32k`。**禁删/禁覆盖/禁重训进此目录**。

## ⏳ 待拍板的开放决策（爸爸定）
1. **E 敏感性章节**：建议写"V-鲁棒性"（CR 对 V 不敏感、落在训练噪声内）——拆掉"为何 V=50 不 V=100"地雷。
2. **headline 训练方差**：0.786 是幸运单训。(A) 维持+披露方差为 limitation / (B) 主配置多训 seed 求 mean±std(~5h) / (C) 其他。

## 红线（不动）
- LyaMAPPO V=50 **32K 金 checkpoint（不可复现，已钉死）** / METHODS 4.1-4.13 / 7 创新 / 定稿超参（BETA=0.02,W_DONE=10,W_HL=2…）
- 共享 DVFS 物理基底（公平性）
- **N=25 原始数据/checkpoint 保留作 N=192 投影证据，绝不删**

## 下一步（待办）
1. **架构图**（若本会话做这个）：2 张**已存在**（M14）——`docs/figures_paper/arch_system_model.png` · `arch_lyamappo_framework.png`；TikZ 投稿源 `docs/tikz/arch_*.tex`（pdflatex 编译过）；出图脚本 `plot_arch_figures.py`；构图清单 `docs/ARCHITECTURE_DIAGRAMS_SPEC.md`。可在此基础上**微调/重做**（连线/配色/标签/布局/中英文）。
2. **论文写作**：EXPERIMENT_CHAPTER 补 E 章节 + intro/method 电池叙事 + GDCO/TD3 baseline 描述段。
3. 先处理上面两个开放决策。

## 复现命令
```bash
cd /home/user/Test1/satellite_mec
# 新容器先装依赖（fresh container 默认没有这些）
pip install numpy torch scipy matplotlib pypdf
# 逐槽序列评估(8策略,~15min) → 出图数据
python eval_series.py
# 出图
python plot_paper_figures.py && python plot_paper_satisfaction.py && \
python plot_pdf_figures.py && python plot_abl_figures.py && python plot_arch_figures.py
# 架构图 TikZ（投稿质量，需 texlive）：
#   apt-get install -y texlive-latex-base texlive-latex-extra texlive-pictures
cd docs/tikz && pdflatex arch_system_model.tex && pdflatex arch_lyamappo_framework.tex && cd ../..
# 训 TD3
python train_td3_sched.py --t_train 16000 --tag 16K
```

---

**结束提示**。先确认已读完上述文档再开干。
