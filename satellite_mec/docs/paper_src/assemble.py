#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""总装 demo3.tex：精确替换 7 处（原文照搬定稿片段）。所有 LaTeX 用 r-string。"""
import sys

path = 'demo3.tex'
with open(path, encoding='utf-8') as f:
    text = f.read()          # 通用换行：CRLF -> LF

reps = []

# ---------- R1 摘要结果句 ----------
reps.append((
r"使神经网络策略在优化过程中隐式逼近理论最优解。基于仿真结果表明，",
r"使神经网络策略在优化过程中隐式逼近理论最优解。仿真结果表明，与 TD3-Sched、MHSPO、GDCO 及本地卸载等基准相比，所提算法在保持第一梯队端到端时延与稳定队列的同时，将系统累计电池健康损耗降至最强基准的约三分之一、系统能耗降至约一半，仅以约 1--5 个百分点的用户满意度为代价，在电池寿命与服务质量之间取得了良好的多目标权衡。"
))

# ---------- R2 引言贡献③ ----------
reps.append((
"""    \\item 我们通过仿真验证了所提方案的有效性。
    仿真结果表明，
""",
"""    \\item 我们通过仿真验证了所提方案的有效性。仿真结果表明，与 TD3-Sched、MHSPO、GDCO 及本地卸载等基准相比，所提算法将系统累计电池健康损耗降至最强基准的约三分之一（较 MHSPO、TD3-Sched 分别降低约 65\\% 与 60\\%）、系统能耗降至约一半，同时维持第一梯队的端到端时延与稳定的队列积压，仅以约 1--5 个百分点的用户满意度为代价；消融实验进一步验证了电池感知与强化学习两个组件缺一不可。
"""
))

# ---------- R3 计算模型 -> Lyapunov-DVFS ----------
reps.append((
r"""采用公平份额调度~\cite{zhang2024mhspo}，
分配给任务$i$的CPU频率为：
\begin{equation}
    f^i_n(t) = \frac{c_n}{N^B_n(t)},
    \label{eq:cpu_alloc}
\end{equation}
其中$c_n$（cycles/s）为卫星$n$的CPU频率，
$N^B_n(t)$为本地处理的任务数量。
时隙$t$内任务$i$的处理数据量为：
\begin{equation}
    r^i_n(t) = \min\!\left\{
    \frac{c_n \cdot \tau}{h^i N^B_n(t)},~
    s^i - \sum_{l=0}^{t-1} r^i_n(l)\right\}。
    \label{eq:workload}
\end{equation}""",
r"""每时隙，卫星$n$为其全部本地任务选择单一计算频率$f^{cmp}_n(t)$，
由每时隙 drift-plus-penalty 子问题
\begin{equation}
    \min_{0 \le f \le c_n}\; V_f\,\kappa f^3\tau - Q^{B}_n(t)\,f\,\tau
\end{equation}
求解，其中$Q^{B}_n(t)$为计算队列剩余CPU周期总量，$c_n$为CPU频率上限。
一阶条件给出闭式解$f^{lyap}_n(t) = \sqrt{Q^{B}_n(t)/(3 V_f \kappa)}$。
为保证截止可行，在公平份额调度（每任务获$f/N^B_n$）下设死线下限
\begin{equation}
    f^{floor}_n(t) = N^B_n(t)\cdot\max_{i \in Q^B_n(t)}
    \frac{\rho^i_n(t)}{\Delta^i_n(t)},
\end{equation}
其中$\rho^i_n(t)$、$\Delta^i_n(t)$分别为任务$i$的剩余CPU周期与剩余时间。
最终计算频率为
\begin{equation}
    f^{cmp}_n(t) = \mathrm{clip}\!\big(\max(f^{lyap}_n(t),\, f^{floor}_n(t)),\, 0,\, c_n\big),
    \label{eq:cpu_alloc}
\end{equation}
权重$V_f$经校准使$f^{cmp}$在满队列工况下恰达$c_n$。
分配给本地任务$i$的频率为$f^{cmp}_n(t)/N^B_n(t)$，时隙$t$内其处理数据量为：
\begin{equation}
    r^i_n(t) = \min\!\left\{
    \frac{f^{cmp}_n(t) \cdot \tau}{h^i N^B_n(t)},~
    s^i - \sum_{l=0}^{t-1} r^i_n(l)\right\}。
    \label{eq:workload}
\end{equation}"""
))

# ---------- R4 计算功耗 / DoD 增量 -> f_cmp ----------
reps.append((
r"""    P^{comp}_n(t) = \sum_{i \in Q^B_n(t)}
    \kappa \left(\frac{c_n}{N^B_n(t)}\right)^3,
    \label{eq:comp_power}""",
r"""    P^{comp}_n(t) = \kappa\,\big(f^{cmp}_n(t)\big)^3,
    \label{eq:comp_power}"""
))
reps.append((
r"""    \Delta DoD^{comp}_n(t) =
    \frac{\tau}{E^{cap}_n}
    \sum_{i \in Q^B_n(t)}
    \kappa \left(\frac{c_n}{N^B_n(t)}\right)^3,
    \label{eq:dod_comp}""",
r"""    \Delta DoD^{comp}_n(t) =
    \frac{\tau}{E^{cap}_n}\,\kappa\,\big(f^{cmp}_n(t)\big)^3,
    \label{eq:dod_comp}"""
))

# ---------- R5a 奖励:基础奖励改写 ----------
reps.append((
r"""对任务$i$执行动作$a$后的即时奖励：

\begin{equation}
r^i_n(t) = -\bar{c}^a_n(i,t),
\label{eq:reward}
\end{equation}""",
r"""对任务$i$执行动作$a$的基础奖励取每时隙 Lyapunov 归一化边费用之负：

\begin{equation}
r^{base,a}_n(i,t) = -\bar{c}^a_n(i,t),
\label{eq:reward}
\end{equation}"""
))

# ---------- R5b 奖励:在 norm2 之后插入 outcome 辅助塑形 ----------
reps.append((
r"""Z_{\max} = T_{snapshot}\cdot\delta_{\max,n}。
\label{eq:norm2}
\end{equation}""",
r"""Z_{\max} = T_{snapshot}\cdot\delta_{\max,n}。
\label{eq:norm2}
\end{equation}

为提升样本效率并稳定训练，在基础奖励上叠加 outcome-aware 辅助塑形项：

\begin{equation}
r^{shape}_n(t) = w_d D_n(t) - w_{to} O_n(t) - w_{re} J_n(t)
- w_h \frac{H_n(t)}{H_{norm}} - w_q P^{Q}_n(t),
\label{eq:reward_shape}
\end{equation}

其中$D_n, O_n, J_n$分别为本时隙完成、超时、拒绝的任务数，$H_n$为健康损耗增量，
$P^Q_n$为队列压力，权重$(w_d, w_{to}, w_{re}, w_h, w_q) = (10, 5, 5, 2, 0.05)$。
卫星$n$在时隙$t$的总奖励为$R_n(t) = \sum_i r^{base,a_i}_n(i,t) + r^{shape}_n(t)$。
塑形项为辅助训练信号，其权重经验选取；理论保证针对基础 Lyapunov 边费用核心成立。"""
))

# ---------- R6 仿真结果整节 ----------
SIM = r"""\subsection{仿真设置}
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
为评估期内系统累计的健康损耗；(3) \emph{系统总能耗}；(4) \emph{端到端时延}，
为每任务 E2E 时延均值；(5) \emph{系统积压}，为系统内待处理任务总数。
满意度与时延为强度量；累计 HL、总能耗与积压为系统级外延量，在 $N=192$ 规模下报告。
% TODO(N-口径)：N=192 系统级量的获取口径（N=25 投影 vs 原生 192）待定。

\subsubsection{训练配置}
LyaMAPPO 采用参数共享 MAPPO，奖励为 Lyapunov 归一化边费用叠加 outcome-aware
辅助塑形（见第~\ref{sec:algorithm}~节）。DVFS 权重 $V_f$ 经校准使 $f^{cmp}$ 在
满队列工况下恰达 $c_n$。训练 32K 时隙；评估在 5400 时隙上进行，
在 10 个独立随机种子上重复，所有结果报告均值 $\pm$ 标准差；对含内部估计量的
启发式（如 MHSPO）评估前设预热期。超参见表~\ref{tab:sim_params}。

\subsection{性能对比与分析}
\label{subsec:comparison}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/queue_backlog.png}
\caption{系统积压量随时间的变化对比}\label{fig:queue}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/satisfaction_pdf.png}
\caption{逐时隙用户满意度分布对比}\label{fig:sat}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/energy_pdf.png}
\caption{逐时隙系统能耗分布对比}\label{fig:energy}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/delay_pdf.png}
\caption{逐时隙系统时延分布对比}\label{fig:delay}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/cumulative_hl.png}
\caption{系统累计健康损失随时间的变化对比}\label{fig:hl}\end{figure}

图~\ref{fig:queue}~至图~\ref{fig:hl}~展示了五种方案的长期系统性能。我们主要关注
五个指标：图~\ref{fig:queue}~的系统积压随时间变化、图~\ref{fig:sat}~的用户满意度、
图~\ref{fig:energy}~的系统能耗开销、图~\ref{fig:delay}~的系统时延，以及
图~\ref{fig:hl}~的累计健康损失随时间变化。需注意，图~\ref{fig:sat}、\ref{fig:energy}、
\ref{fig:delay}~为系统指标在不同时隙上的概率密度分布。

图~\ref{fig:queue}~展示系统积压随时间的变化。所有策略的队列在整个轨道周期内均平稳、
不发散：TD3-Sched 积压最低，LyaMAPPO 与 MHSPO 基本持平，二者凭借 Lyapunov 队列漂移
最小化机制将积压控制在稳定水平，明显低于卸载能力薄弱的 GDCO 与 LSO。

图~\ref{fig:sat}~与图~\ref{fig:delay}~分别对比 LyaMAPPO 与四个基线在卫星边缘网络
场景下的逐时隙用户满意度与端到端时延。LyaMAPPO 与 TD3-Sched 在时延上同处第一梯队，
明显优于 MHSPO、GDCO 与 LSO；这得益于任务级顺序决策等机制，使 LyaMAPPO 的热点卫星
队列峰值与 TD3 并列最低。在满意度上，TD3-Sched 略占优势，是因为其代价函数不含电池
健康与 DoD 约束，能够采取更激进的高时钟卸载策略；但正如下文所示，这一优势以牺牲系统
能耗与电池寿命为代价。相比之下，缺乏有效卸载与队列控制的 LSO 满意度最低，服务能力
几乎丧失。

图~\ref{fig:energy}~与图~\ref{fig:hl}~进一步对比系统能耗与累计电池健康损失（HL）。
LyaMAPPO 在各对比策略中均展现优势：其系统总能耗约为 TD3 的 57\%、MHSPO 的 54\%，
累计 HL 约为 MHSPO 的三分之一。这得益于 Lyapunov--DVFS 耦合下的低时钟运行，以及放电
深度虚拟队列 $z_n$ 所实现的耗能``择时''。相反，缺乏电池感知的算法电池损耗严重：
MHSPO 与 TD3-Sched 的累计 HL 分别增至 LyaMAPPO 的 2.9 倍与 2.5 倍，系统能耗亦分别
高出约 86\% 与 76\%。对累计 HL 的三因子分解表明，这种恶化主要并非源于更深的放电
（各策略损伤率水平相近），而是源于它们将耗能决策集中于高放电深度时段。值得注意的是，
LSO 不具备卸载能力，其过载卫星须在本地处理全部到达任务，致使热点卫星长期处于深度放电
状态，且能量消耗多集中于高放电深度时段；尽管其系统总能耗为各策略中最低，但该结果源于
大量任务因过载而超时，并非高效运行所致，其累计健康损失仍与 LyaMAPPO 相当。这表明，
系统总能耗与累计健康损失之间并非单调对应，决定健康损失的关键在于能量消耗所处的放电
深度及其空间分布，而非能耗总量。

综上，LyaMAPPO 实现了优异的多目标权衡：以约 1--5 个百分点的满意度让步，换取显著更低
的系统能耗与电池健康损耗，同时维持第一梯队的时延与稳定的队列。其根源在于
Lyapunov--MAPPO 深度耦合与 $z_n$ 电池感知择时机制——将长期电池约束转化为可观测的
虚拟队列，并通过 RL 长视野学习，使能量消耗集中于放电深度较低的时段，从而以更低的总
能量完成计算。

\subsection{消融实验与分析}
\label{subsec:ablation}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ablation_satisfaction.png}
\caption{逐时隙用户满意度分布消融}\label{fig:abl_sat}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/abl_delay_pdf.png}
\caption{逐时隙系统时延分布消融}\label{fig:abl_delay}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ablation_energy.png}
\caption{逐时隙系统能耗分布消融}\label{fig:abl_energy}\end{figure}
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ablation_cumulative_hl.png}
\caption{系统累计健康损失随时间的变化消融}\label{fig:abl_hl}\end{figure}

\begin{table}[t]\centering
\caption{消融实验结果（$N=192$，均值 $\pm$ 标准差，10 种子）}
\label{tab:ablation}
\begin{tabular}{lcccc}
\hline
变体 & 用户满意度 & 时延(s) & 累计 HL & 总能耗(kJ) \\
\hline
LyaMAPPO（完整） & $0.787\pm0.001$ & $3.33\pm0.01$ & $183\pm4$ & $6957\pm61$ \\
MAPPO-NoDoD & $0.824\pm0.001$ & $3.09\pm0.01$ & $554\pm6$ & $13238\pm92$ \\
LyapunovGreedy & $0.425\pm0.002$ & $6.25\pm0.02$ & $70\pm2$ & $3356\pm46$ \\
\hline
\end{tabular}
\end{table}

为厘清 LyaMAPPO 中关键组件的作用，本节分别剥离两个组件：电池感知（放电深度虚拟队列
$z_n$ 与健康损失奖励项 $w_h$）与强化学习。移除前者得到 MAPPO-NoDoD（$z_n$ 与 $w_h$
一并置零，为合并消融，用于界定电池感知整体的作用）；移除后者得到 LyapunovGreedy
（保留同一 Lyapunov 代价计算，但以无学习的贪心策略逐任务最小化当前时隙代价）。
图~\ref{fig:abl_sat}~至图~\ref{fig:abl_hl}~及表~\ref{tab:ablation}~给出两变体与完整
模型在用户满意度、系统时延、系统能耗与累计健康损失上的对比。

图~\ref{fig:abl_sat}~与图~\ref{fig:abl_delay}~分别对比用户满意度与端到端时延。移除
电池约束后，MAPPO-NoDoD 的满意度不降反升至 0.824（较完整模型高约 4 个百分点），时延
亦略降至 3.09~s；这是因为放开电池约束后，策略可采取更激进的高时钟卸载。该结果从反面
量化了完整模型为保护电池而主动让出的吞吐（约 4 个百分点）。与之相反，LyapunovGreedy
的满意度骤降至 0.425（约为完整模型的一半），时延显著恶化至 6.25~s。其原因在于，缺乏
强化学习的长视野信用分配后，贪心策略仅能最小化当前时隙代价，无法进行跨时隙的任务取舍，
导致大量任务超时、队列积压加剧。这表明，Lyapunov 机制本身虽能维持队列稳定，但须与
强化学习的长视野决策相结合，方能在均衡负载的同时保持高吞吐。

图~\ref{fig:abl_energy}~与图~\ref{fig:abl_hl}~分别对比系统能耗与累计健康损失。移除
电池感知后，MAPPO-NoDoD 的累计 HL 急剧上升至完整模型的 3.0 倍，系统能耗上升至 1.9 倍；
对累计 HL 的三因子分解表明，这一恶化同时来自更差的耗能择时与更高的总能耗，而非更深的
放电，印证 $z_n$ 与 $w_h$ 共同构成抑制健康损失的关键机制。LyapunovGreedy 的累计 HL
虽更低（约为完整模型的 0.38 倍），但该结果并非源于有效的电池管理：其贪心策略实际处理的
任务量极少（满意度仅 0.425），健康损失的降低是服务能力严重退化的伴随现象，而非可用的
优化结果。

综上，两个消融变体沿相反方向失效：移除电池感知（MAPPO-NoDoD）导致电池健康损失激增，
移除强化学习（LyapunovGreedy）导致吞吐性能崩溃。唯有同时具备电池感知择时（$z_n$ 与
$w_h$）与强化学习长视野决策的完整 LyaMAPPO，方能在电池健康与服务质量之间取得可用的
平衡。这一结果验证了本文两项核心设计的不可或缺性。"""

reps.append((
r"""\label{sec:simulation}
% ============================================================

（待补充）""",
r"""\label{sec:simulation}
% ============================================================

""" + SIM
))

# ---------- R7 结论 ----------
CONC = r"""本文研究了 LEO 卫星边缘计算网络中电池感知的多跳任务卸载问题。针对深度放电加速电池
容量衰减、威胁卫星在轨寿命这一核心挑战，本文采用半指数函数量化放电深度与电池寿命
损耗的非线性关系，通过定义 DoD 虚拟队列 $z_n$ 并结合一阶泰勒线性化，在 Lyapunov
漂移加惩罚框架内将非线性电池约束转化为可优化的线性代理目标；进而设计 Lyapunov
理论塑形奖励驱动的参数共享 MAPPO 分布式在线调度算法，使神经网络策略在优化过程中
隐式逼近理论最优解，并实现训练复杂度与卫星规模解耦。

仿真结果表明，与 TD3-Sched、MHSPO、GDCO 及本地卸载等基准相比，所提算法在保持第一
梯队端到端时延与稳定队列积压的同时，将系统累计电池健康损耗降至最强基准的约三分之一、
系统能耗降至约一半，仅以约 1--5 个百分点的用户满意度为代价。机理分析进一步揭示，这一
电池优势主要源于 $z_n$ 驱动的耗能``择时''——将能量消耗调度至放电深度较低的时段，
从而在损耗函数的强凸性下换取指数级的健康收益；消融实验验证了电池感知与强化学习两个
组件缺一不可。

本文仍存在若干局限，有待后续研究：其一，主结果基于单一训练所得策略，训练过程本身的
方差有待通过多次独立训练进一步刻画；其二，DVFS 与电池模型均为模型级抽象，尚未纳入
硬件在环验证；其三，本文聚焦单服务卸载场景，未涉及资源切片等多服务扩展。未来工作将
围绕多服务资源联合优化、硬件在环验证与更大规模原生评估展开。"""

reps.append((
r"""\label{sec:conclusion}
% ============================================================

（待补充）""",
r"""\label{sec:conclusion}
% ============================================================

""" + CONC
))

# ---------- 应用 + 断言 ----------
for i, (old, new) in enumerate(reps, 1):
    cnt = text.count(old)
    if cnt != 1:
        print(f"[FAIL] R{i}: 匹配 {cnt} 次（应为 1）", file=sys.stderr)
        sys.exit(1)
    text = text.replace(old, new, 1)
    print(f"[OK] R{i} 替换成功")

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)
print("总装完成 ->", path)
