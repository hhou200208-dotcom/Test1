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
satellite_mec/docs/ARCHITECTURE_DIAGRAMS_SPEC.md  # 架构图构图清单（下一步用）
satellite_mec/docs/figures_paper/          # 14 张论文图（已做好）
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

## 红线（不动）
- LyaMAPPO V=50 32K checkpoint / METHODS 4.1-4.13 / 7 创新 / 定稿超参（BETA=0.02,W_DONE=10,W_HL=2…）
- 共享 DVFS 物理基底（公平性）
- **N=25 原始数据/checkpoint 保留作 N=192 投影证据，绝不删**

## 下一步（待办）
1. **论文写作**：experiment 章节（图都齐，**含 2 张架构图**）、GDCO/TD3 baseline 描述段、电池叙事写进 intro/method
2. 架构图微调（按需）：连线/配色/标签可再调；TikZ 已编译通过，可直接 `\includegraphics` 进论文
3. 可选：满意度时间曲线、训练收敛曲线

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
