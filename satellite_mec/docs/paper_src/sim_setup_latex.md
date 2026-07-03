# 仿真结果 · 1.1 仿真设置（LaTeX，待并入 demo3.tex 的 sec:simulation）

> 归因收敛：仅用论文真有的机制（DoD 虚拟队列 z_n、Lyapunov-DVFS、outcome 辅助塑形、参数共享 MAPPO）。
> 诚实：N=192 系统级量口径句留 `%TODO` 占位（投影问题搁置）。
> cite：MHSPO=\cite{zhang2024mhspo}（已在 tex）；TD3/GDCO 的 key 待与 bib 对齐（下用占位）。

```latex
\subsection{仿真设置}
\label{subsec:sim_setup}

\subsubsection{场景与参数}
我们在 Walker 星座上仿真一个卫星边缘计算网络，星座轨道高度 550~km，
共 $N=192$ 颗卫星（16 个轨道平面 $\times$ 12 颗）。仿真以时隙长度 $\tau=1$~s
逐槽推进，运行时长为一个轨道周期（5400 时隙）。为刻画空间负载不均衡与由此
产生的电池压力，$1/5$ 的卫星为高负载星，任务到达率 $\lambda_{\text{high}}=4.0$
（任务/时隙），其余为低负载星 $\lambda_{\text{low}}=0.1$，到达均服从泊松过程。
电池与健康损耗模型（半指数寿命损耗、DoD 动态与 DoD 虚拟队列 $z_n$）见
第~\ref{sec:system_model}~节；主要参数汇总于表~\ref{tab:sim_params}。
本文所提算法在下文记为 \textbf{LyaMAPPO}。

\begin{table}[t]
\centering
\caption{主要仿真参数}
\label{tab:sim_params}
\begin{tabular}{ll}
\hline
参数 & 取值 \\
\hline
卫星数 $N$ & 192（16 $\times$ 12 Walker）\\
轨道高度 / 周期 & 550~km / 5400 时隙 \\
时隙长度 $\tau$ & 1~s \\
高/低负载率 $\lambda$ & 4.0 / 0.1（1/5 为高负载）\\
任务大小 $s^i$ & 10--50~Mb \\
计算密度 $h^i$ & 10--30~cycles/bit~\cite{li_tsc_2024} \\
截止时间 & 1--12~s \\
最大转发跳数 & 3 \\
CPU 频率上限 $c_n$ & 2~GHz~\cite{zhang_tmc_2023} \\
能耗系数 $\kappa$ & $1\times10^{-26}$~\cite{zhang_tmc_2023} \\
单星并发上限 & 6 \\
ISL 速率 & 100--300~Mb/s \\
发射功率 & 0.1~W \\
每星邻居数 & 4 \\
电池容量 $E^{cap}_n$ & 36~kJ \\
太阳能功率 / 日照比 & $\le 30$~W / 0.65 \\
DoD 范围 & $[0.1,\,0.8]$ \\
损耗曲线系数 $a$ & 0.8 \\
Lyapunov $V$ / $\eta$ & 50 / 0.5 \\
奖励权重 $(w_d,w_{to},w_{re},w_h,w_q)$ & $(10,5,5,2,0.05)$ \\
PPO $\gamma$/$\lambda_{\text{GAE}}$/$\epsilon$/熵 & 0.99 / 0.95 / 0.2 / 0.02 \\
学习率（actor/critic） & $10^{-4}$ / $10^{-3}$ \\
minibatch / epoch / rollout & 64 / 2 / 64 \\
训练 / 评估时长 & 32K / 5400 时隙 \\
评估种子数 & 10 \\
\hline
\end{tabular}
\end{table}

\subsubsection{对比算法与消融变体}
我们将 LyaMAPPO 与四个基准对比，四者均运行在同一物理基底（相同的
DVFS 频率选择、电池与 ISL 模型）之上，差异仅来自卸载/调度策略：
\begin{itemize}
    \item \textbf{TD3-Sched}~\cite{huang2024td3}：学习型基准，取自双时间尺度调度
    框架中的 TD3 调度器，保留 clipped double-Q、延迟策略更新与目标策略平滑；
    其代价函数不含电池健康与 DoD 约束。
    \item \textbf{MHSPO}~\cite{zhang2024mhspo}：将多跳星间对等卸载与 Lyapunov
    队列控制相结合的最强经典基准，以延迟在线梯度下降在线预测邻居负载并规避拥塞。
    \item \textbf{GDCO}~\cite{chen2025gdco}：基于精确势博弈的分布式卸载，非学习型，
    各卫星依博弈均衡决策，仅优化能耗、不含电池健康项。
    \item \textbf{LSO}：平凡下界，所有任务在接入卫星本地处理、不启用星间卸载。
\end{itemize}
此外设置两个消融变体：\textbf{MAPPO-NoDoD}（移除电池感知，即将 $z_n$ 与
健康损耗奖励项一并置零，为合并消融）与 \textbf{LyapunovGreedy}（移除强化学习，
以无学习的贪心逐任务最小化当前时隙代价）。

\subsubsection{评估指标}
我们报告五个指标：(1) \emph{用户满意度}（主指标）$=$ 满足截止时间的任务数
与（完成 $+$ 超时）任务数之比；(2) \emph{电池累计健康损失}（HL），
为评估期内系统累计的健康损耗（式~\ref{eq:hl_increment}）；
(3) \emph{系统总能耗}；(4) \emph{端到端时延}，为每任务 E2E 时延均值；
(5) \emph{系统积压}，为系统内待处理任务总数。满意度与时延为强度量；
累计 HL、总能耗与积压为系统级外延量，在 $N=192$ 规模下报告。
% TODO(N-口径)：N=192 系统级量的获取口径（N=25 投影 vs 原生 192）待定，见 method_stage0_revisions.md

\subsubsection{训练配置}
LyaMAPPO 采用参数共享 MAPPO，奖励为 Lyapunov 归一化边费用叠加 outcome-aware
辅助塑形（见第~\ref{sec:algorithm}~节）。DVFS 权重 $V_f$ 经校准使 $f^{cmp}$ 在
满队列工况下恰达 $c_n$。训练 32K 时隙；评估在 5400 时隙上进行，
在 10 个独立随机种子上重复，所有结果报告均值 $\pm$ 标准差；对含内部估计量的
启发式（如 MHSPO）评估前设预热期。超参见表~\ref{tab:sim_params}。
```
```

## 待您/后续对齐的占位
- `\cite{huang2024td3}`、`\cite{chen2025gdco}`：TD3、GDCO 的 bib key 需与您 demo3.tex 参考文献表对齐（MHSPO 已用 `zhang2024mhspo`）。
- `\ref{eq:hl_increment}`：指向方法节的健康损耗增量式，实际标签名以 tex 为准。
- `%TODO(N-口径)`：投影句搁置，占位不写错。
