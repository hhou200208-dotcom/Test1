# 方法节 阶段0 一致性修订（对齐代码/实测）— LaTeX 待并入 demo3.tex

> 依据代码核实：`satellite.py` 用 `core.dvfs.select_freq`（DVFS）；`env.py:224/248-258` 注入 outcome 塑形。
> 论文 demo3.tex 现状：计算模型是定频公平份额（eq:cpu_alloc/comp_power/dod_comp）、奖励是纯 Lyapunov 边费用（eq:reward）。
> **决定（已定稿）**：① 能耗模型改 DVFS（下 §1，采用）；② 奖励 **采用版本 A**（披露辅助塑形，见下 §2）；③ N=25→192 投影：本阶段搁置，写设置节时再议。

## 1 计算/能耗模型 → Lyapunov-DVFS（替换 eq:cpu_alloc / comp_power / dod_comp）

```latex
\subsubsection{计算模型（Lyapunov-DVFS）}
每时隙，卫星 $n$ 为其全部本地任务选择单一计算频率 $f^{cmp}_n(t)$，
由每时隙 drift-plus-penalty 子问题
\begin{equation}
\min_{0\le f\le c_n}\; V_f\,\kappa f^3\tau \;-\; Q^{B}_n(t)\,f\,\tau
\end{equation}
求解，其中 $Q^{B}_n(t)$ 为计算队列剩余 CPU 周期总量，$c_n$ 为 CPU 频率上限。
一阶条件给出闭式解 $f^{lyap}_n(t)=\sqrt{Q^{B}_n(t)/(3V_f\kappa)}$。
为保证截止可行，在公平份额调度（每任务获 $f/N^B_n$）下设死线下限
\begin{equation}
f^{floor}_n(t)=N^B_n(t)\cdot\max_{i\in Q^B_n(t)}
\frac{\text{剩余周期}_i}{\text{剩余时间}_i},
\end{equation}
最终频率 $f^{cmp}_n(t)=\mathrm{clip}\!\big(\max(f^{lyap}_n,\,f^{floor}_n),\,0,\,c_n\big)$；
$V_f$ 经校准使 $f^{cmp}$ 在满队列工况下恰达 $c_n$。
分配给任务 $i$ 的频率为 $f^{cmp}_n(t)/N^B_n(t)$。
\end{equation}
```

能耗与 DoD 增量（替换 comp_power / dod_comp）：
```latex
P^{comp}_n(t)=\kappa\,\big(f^{cmp}_n(t)\big)^3,\qquad
\Delta DoD^{comp}_n(t)=\frac{\tau}{E^{cap}_n}\,\kappa\,\big(f^{cmp}_n(t)\big)^3.
```

## 2 奖励函数 —— 两版待选

### 版本 A（如实披露辅助塑形）✅ 已选定
```latex
\subsubsection{奖励函数}
基础奖励为每时隙 Lyapunov 归一化边费用之负（式\ref{eq:cost_local}/\ref{eq:cost_offload}）：
$r^{base}_n(t)=-\sum_i \bar{c}^{a_i}_n(i,t)$。
为提升样本效率并稳定训练，叠加 outcome-aware 辅助塑形：
\begin{equation}
r^{shape}_n(t)=w_d D_n(t)-w_{to}O_n(t)-w_{re}J_n(t)
-w_h \tfrac{H_n(t)}{H_{norm}}-w_q P^{Q}_n(t),
\end{equation}
其中 $D_n,O_n,J_n$ 为本时隙完成/超时/拒绝任务数，$H_n$ 为健康损耗增量，
$P^Q_n$ 为队列压力，权重 $(w_d,w_{to},w_{re},w_h,w_q)=(10,5,5,2,0.05)$。
总奖励 $R_n(t)=r^{base}_n(t)+r^{shape}_n(t)$。塑形项为辅助训练信号，
其权重经验选取；理论保证（定理\ref{thm:perf}）针对基础 Lyapunov 边费用核心成立。
```

### 版本 B（维持纯 Lyapunov 奖励，与论文现状一致）
```latex
\subsubsection{奖励函数}
奖励取每时隙 Lyapunov 归一化边费用之负 $r^i_n(t)=-\bar{c}^{a}_n(i,t)$
（式\ref{eq:cost_local}/\ref{eq:cost_offload}），三项分别对应队列漂移、
健康损耗与 DoD 虚拟队列惩罚。由此最小化累积奖励等价于逐时隙逼近
Lyapunov 子问题最优解，构成定理\ref{thm:perf} 的依据。
```

### 取舍
| | A（披露塑形） | B（纯 Lyapunov） |
|---|---|---|
| 与实测一致 | ✅ | ❌ 漏 W_done 等 |
| 复现性 | ✅ | ⚠️ 纯边费用复现不出报告数字 |
| 理论叙事 | 🟡 "逼近最优"限定于核心 | ✅ 最干净 |
| 诚实红线 | ✅ | ⚠️ 方法≠实测 |

建议：版本 A（复现性/一致性为 CCF-A 硬指标；塑形写成辅助信号+理论针对核心是标准做法）。

> 注：权重用定稿值 (10,5,5,2,0.05)（≠ config 默认，见 checkpoints/README）；另有 COMPLETION_BONUS=1（可并入 $w_d$ 叙述或单列）。
