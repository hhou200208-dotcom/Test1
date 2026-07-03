# 仿真结果 · 1.2 性能对比 + 1.3 消融（LaTeX，直接转自定稿中文，待并入 demo3.tex sec:simulation）

> 直接转 section2_perf_final.md / section3_ablation_final.md，原文照搬，仅 LaTeX 化。
> 图文件在 docs/figures_paper/（图7 时延消融用 figures_clean/abl_delay_pdf）。graphicspath 以工程为准。
> %TODO：图8 原 docx 标题笔误"系统延迟"→已按数据改为"系统能耗"。

```latex
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
平衡。这一结果验证了本文两项核心设计的不可或缺性。
```

## 转换说明
- 定稿中文**原文照搬**，仅做 LaTeX 化：`%`→`\%`、`—`→`——`/`--`、中文引号→```` ``''````、`z_n/w_h`→数学模式、图号→`\ref`、加图/表环境。
- W_HL 用方法节符号 `$w_h$`（与奖励式一致）。
- 图8 标题按数据改为"系统能耗"（原 docx 笔误已纠）。
- `figures/` 路径与 `\graphicspath` 以您 tex 工程为准。
- ⚠️ 一处待您定：图~\ref{fig:sat}/\ref{fig:delay}~段仍含"**任务级顺序决策**"（act_one）——它不在论文贡献里。要严格归因收敛可改为"得益于负载均衡与 DVFS 截止下限"；本次按"直接转"保留原文，您一句话我就换。
