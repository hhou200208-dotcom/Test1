"""Architecture figures for the LyaMAPPO paper (matplotlib renderer).

Produces two paper-quality schematic figures, parameters aligned to the code
(verified values: Actor 54-d, Critic 245-d, V=50, eta=0.5, finalized reward
weights 10/5/5/2/0.05, beta_task=0.5, DVFS f*=sqrt(Q/3*V_DVFS*kappa)):

  docs/figures_paper/arch_system_model.png       (Fig. system model)
  docs/figures_paper/arch_lyamappo_framework.png (Fig. LyaMAPPO framework, CTDE)

Spec: docs/ARCHITECTURE_DIAGRAMS_SPEC.md. Colors echo the result figures
(LyaMAPPO red #d62728). Paper scenario: N=192 LEO (16x12 Walker).
A TikZ submission source of the same two figures lives in docs/tikz/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle

OUT = os.path.join(os.path.dirname(__file__), 'docs', 'figures_paper')
os.makedirs(OUT, exist_ok=True)

# ── palette (env=gray-blue, Lyapunov=orange, Actor=red, Critic=green) ──
C_ENV, C_ENV_E   = '#dce6f2', '#3f6090'
C_LYA, C_LYA_E   = '#fde7cf', '#d97f0c'
C_ACT, C_ACT_E   = '#f7d3d3', '#d62728'
C_CRI, C_CRI_E   = '#d4ecd4', '#2ca02c'
C_QUE, C_QUE_E   = '#eef2f7', '#7f8c9b'
C_TASK, C_TASK_E = '#fff4cc', '#b58a00'  # task = soft yellow
C_DONE           = '#cfe9cf'
STAR             = '#b8860b'
TASK_FLOW, ENE_FLOW, INFO_FLOW = '#222222', '#e8820c', '#8a8a8a'


# ── language / fonts (zh = WenQuanYi Zen Hei; math symbols stay LaTeX) ──
def set_lang_font(lang):
    if lang == 'zh':
        matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


# Figure-1 text table. Variable symbols (Q^F_n, V=50, ...) are universal and
# left in math mode; only descriptive labels are translated.
T_SYS = {
    'en': {
        'title': 'System Model — LEO Satellite Edge Computing with Battery-Health-Aware Offloading',
        'a_label': '(a) LEO constellation & ISL',
        'a_note': ('N = 192  (16$\\times$12 Walker);  4$\\times$4 shown ($\\cdots\\times$N)\n'
                   'ISL: rate $R_{n,m}$, prop. delay $\\tau_{n,m}$  ($B$=100–300 Mb/s)\n'
                   'red = high-load $\\lambda_{hi}{=}4.0$ · blue = low-load $\\lambda_{lo}{=}0.1$'),
        'b_label': '(b) On-board task pipeline (one satellite)',
        'task_arr': 'Task arrivals — Poisson $\\lambda$',
        'task_note': 'task $K_i=\\{D_i$ bits, $X_i$ cyc/bit, deadline$\\}$,  $D_i\\in$[10,50] Mb',
        'fwd_q': 'Forward queue  $Q^F_n$  (bits)',
        'decision': 'offload decision\n$a\\in\\{0,1,\\dots,4\\}$',
        'cmp_q': 'Compute queue\n$Q^B_n$ (bits)',
        'dvfs_cpu': 'DVFS CPU  $f_{cmp}$',
        'done': 'Done $\\checkmark$',
        'local': '$a{=}0$\nlocal',
        'isl_box': 'ISL →\nneighbor $m_k$',
        'forward': '$a{=}k$ forward',
        'to_neighbor': "→ neighbor's $Q^F_n$",
        'dvfs_head': 'DVFS closed-form (Li TSC’24):',
        'dvfs_eq': ('$f_{cmp}=\\min(F_{max},\\max(\\sqrt{Q^B/3V_{DVFS}\\kappa},f_{floor}))$\n'
                    '$F_{max}{=}2$ GHz,  $E_{cmp}{=}\\kappa f^3\\tau$'),
        'c_label': '(c) Battery energy cycle',
        'solar': 'Solar harvest\n$P_{solar}\\leq 30$ W (65% lit)',
        'battery': 'Battery  $E_{cap}{=}36$ kJ\nDoD $\\delta_n\\in[0.1,0.8]$',
        'discharge': 'Discharge:\ncompute $\\kappa f^3\\tau$ + transmit $P_T D/R$',
        'health': 'Health loss\n$H_n=\\delta_n\\,10^{\\,a(\\delta_n-1)}$\n(irreversible — paper headline)',
        'lg_task': 'task flow', 'lg_ene': 'energy flow', 'lg_info': 'state broadcast (4 neighbours)',
    },
    'zh': {
        'title': '系统模型 —— 面向电池健康的 LEO 卫星边缘计算任务卸载',
        'a_label': '(a) LEO 星座与星间链路 (ISL)',
        'a_note': ('$N=192$（16$\\times$12 Walker）；示意 4$\\times$4（$\\cdots\\times N$）\n'
                   'ISL：速率 $R_{n,m}$，传播时延 $\\tau_{n,m}$（$B$=100–300 Mb/s）\n'
                   '红 = 高负载 $\\lambda_{hi}{=}4.0$ · 蓝 = 低负载 $\\lambda_{lo}{=}0.1$'),
        'b_label': '(b) 星上任务流水线（单颗卫星）',
        'task_arr': '任务到达 —— 泊松过程 $\\lambda$',
        'task_note': '任务 $K_i=\\{D_i$ 比特, $X_i$ 周期/比特, 截止$\\}$，$D_i\\in$[10,50] Mb',
        'fwd_q': '前向队列  $Q^F_n$（比特）',
        'decision': '卸载决策\n$a\\in\\{0,1,\\dots,4\\}$',
        'cmp_q': '计算队列\n$Q^B_n$（比特）',
        'dvfs_cpu': 'DVFS 处理器  $f_{cmp}$',
        'done': '完成 $\\checkmark$',
        'local': '$a{=}0$\n本地',
        'isl_box': 'ISL →\n邻居 $m_k$',
        'forward': '$a{=}k$ 转发',
        'to_neighbor': '→ 邻居的 $Q^F_n$',
        'dvfs_head': 'DVFS 闭式解 (Li TSC’24)：',
        'dvfs_eq': ('$f_{cmp}=\\min(F_{max},\\max(\\sqrt{Q^B/3V_{DVFS}\\kappa},f_{floor}))$\n'
                    '$F_{max}{=}2$ GHz，$E_{cmp}{=}\\kappa f^3\\tau$'),
        'c_label': '(c) 电池能量循环',
        'solar': '太阳能采集\n$P_{solar}\\leq 30$ W（65% 光照）',
        'battery': '电池  $E_{cap}{=}36$ kJ\n放电深度 $\\delta_n\\in[0.1,0.8]$',
        'discharge': '放电：\n计算 $\\kappa f^3\\tau$ + 传输 $P_T D/R$',
        'health': '健康损耗\n$H_n=\\delta_n\\,10^{\\,a(\\delta_n-1)}$\n（不可逆 —— 论文核心命题）',
        'lg_task': '任务流', 'lg_ene': '能量流', 'lg_info': '状态广播（4 邻居）',
    },
}


# ── primitives ────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text='', fc='#fff', ec='#333', lw=1.4, fs=10,
        tc='#111', bold=False, rounded=True, zorder=2, va='center', pad=0.18):
    bs = f"round,pad=0,rounding_size={0.10 if rounded else 0.0}"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=bs, fc=fc, ec=ec,
                                lw=lw, zorder=zorder, mutation_aspect=1))
    if text:
        ty = y + h / 2 if va == 'center' else (y + h - pad if va == 'top' else y + pad)
        ax.text(x + w / 2, ty, text, ha='center', va=va, fontsize=fs,
                color=tc, fontweight='bold' if bold else 'normal', zorder=zorder + 1)
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, text='', fc=C_ENV, ec='#333', lw=1.4, fs=9, zorder=3):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=zorder))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, zorder=zorder + 1)


def arrow(ax, p0, p1, color=TASK_FLOW, ls='-', lw=1.7, style='-|>', rad=0.0,
          ms=15, zorder=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                                 linestyle=ls, mutation_scale=ms, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def lbl(ax, x, y, text, fs=8.5, color='#222', style='normal', ha='center',
        va='center', bg=None, zorder=6, weight='normal'):
    bbox = dict(boxstyle='round,pad=0.18', fc=bg, ec='none', alpha=0.92) if bg else None
    ax.text(x, y, text, fontsize=fs, color=color, style=style, ha=ha, va=va,
            zorder=zorder, bbox=bbox, fontweight=weight)


def star(ax, x, y, n, fs=12):
    ax.text(x, y, f'★{n}', color=STAR, fontsize=fs, fontweight='bold',
            ha='center', va='center', zorder=8)


def sat_icon(ax, cx, cy, s=0.16, body=C_ENV, ec=C_ENV_E, panel='#33527a', zorder=3):
    ax.add_patch(Rectangle((cx - s, cy - s * 0.62), 2 * s, s * 1.24, fc=body,
                           ec=ec, lw=1.0, zorder=zorder + 1))
    for sgn in (-1, 1):
        ax.add_patch(Rectangle((cx + sgn * s * 1.05 - (s * 0.95 if sgn > 0 else 0),
                                cy - s * 0.42), s * 0.95, s * 0.84, fc=panel,
                               ec='#1d2c44', lw=0.6, zorder=zorder))


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — System Model
# ══════════════════════════════════════════════════════════════════════
def fig_system_model(lang='en'):
    set_lang_font(lang)
    t = T_SYS[lang]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    ax.text(8, 8.66, t['title'], ha='center', fontsize=15, fontweight='bold')

    # ---- region (a): constellation + ISL + solar ----
    box(ax, 0.3, 1.05, 5.0, 7.15, fc='#fbfcfe', ec='#c4cdda', lw=1.1, rounded=True)
    lbl(ax, 2.8, 7.95, t['a_label'], fs=11, weight='bold')
    gx = np.linspace(1.15, 4.45, 4); gy = np.linspace(2.35, 6.7, 4)
    hi = {(1, 2), (2, 1)}  # representative high-load satellites
    pos = {}
    for i, yv in enumerate(gy):
        for j, xv in enumerate(gx):
            pos[(i, j)] = (xv, yv)
    # ISL links (4-neighbour grid) — #1: brighter & thicker so topology reads
    for (i, j), (xv, yv) in pos.items():
        for di, dj in ((0, 1), (1, 0)):
            if (i + di, j + dj) in pos:
                x2, y2 = pos[(i + di, j + dj)]
                ax.plot([xv, x2], [yv, y2], color=C_ENV_E, lw=1.7, alpha=0.9, zorder=1)
    for (i, j), (xv, yv) in pos.items():
        high = (i, j) in hi
        sat_icon(ax, xv, yv, s=0.17, body=('#f6d2d2' if high else C_ENV),
                 ec=(C_ACT_E if high else C_ENV_E), panel=('#9e2a2a' if high else '#33527a'))
    # solar + eclipse
    ax.add_patch(Circle((0.95, 7.35), 0.30, fc='#ffd24d', ec='#e0a800', lw=1.0, zorder=2))
    for ang in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([0.95 + 0.34 * np.cos(ang), 0.95 + 0.5 * np.cos(ang)],
                [7.35 + 0.34 * np.sin(ang), 7.35 + 0.5 * np.sin(ang)],
                color='#e0a800', lw=1.3, zorder=2)
    arrow(ax, (1.25, 7.1), (pos[(2, 0)][0] - 0.05, pos[(2, 0)][1] + 0.25),
          color=ENE_FLOW, ls=(0, (4, 2)), lw=1.5)
    lbl(ax, 2.66, 1.86, t['a_note'], fs=8.2, style='italic', color='#444')
    # zoom callout to region (b)
    for cyc in (0.55, -0.55):
        ax.plot([pos[(1, 2)][0] + 0.18, 5.65], [pos[(1, 2)][1] + cyc * 0.5, 7.6 if cyc > 0 else 1.55],
                color='#9aa6b6', lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.add_patch(Circle(pos[(1, 2)], 0.30, fill=False, ec=C_ACT_E, lw=1.6, ls=':', zorder=4))

    # ---- region (b): on-board pipeline ----
    box(ax, 5.65, 1.05, 6.05, 7.15, fc='#ffffff', ec='#c4cdda', lw=1.1)
    lbl(ax, 8.67, 7.95, t['b_label'], fs=11, weight='bold')
    box(ax, 6.55, 6.95, 4.25, 0.8, t['task_arr'], fc=C_TASK, ec=C_TASK_E, fs=10)
    lbl(ax, 8.67, 6.62, t['task_note'], fs=7.6, color='#555')
    box(ax, 6.55, 5.55, 4.25, 0.78, t['fwd_q'], fc=C_QUE, ec=C_QUE_E, fs=10)
    diamond(ax, 8.67, 4.42, 2.7, 1.05, t['decision'], fc=C_ENV, fs=9)
    box(ax, 6.05, 2.55, 2.55, 0.78, t['cmp_q'], fc=C_QUE, ec=C_QUE_E, fs=9)
    box(ax, 6.05, 1.35, 2.55, 0.82, t['dvfs_cpu'], fc='#e9eef5', ec='#5b6b80', fs=9.5, bold=True)
    box(ax, 9.05, 1.35, 1.75, 0.82, t['done'], fc=C_DONE, ec=C_CRI_E, fs=10, bold=True)
    # arrows (task flow)
    arrow(ax, (8.67, 6.95), (8.67, 6.33))
    arrow(ax, (8.67, 5.55), (8.67, 4.96))
    arrow(ax, (7.9, 4.05), (7.32, 3.33)); lbl(ax, 7.12, 3.62, t['local'], fs=8)
    arrow(ax, (7.32, 2.55), (7.32, 2.17))
    arrow(ax, (8.6, 1.76), (9.05, 1.76))
    # forward branch -> ISL out
    arrow(ax, (9.95, 4.05), (10.95, 5.05), rad=-0.2)
    box(ax, 10.05, 5.05, 1.55, 0.72, t['isl_box'], fc='#eaf0fa', ec=C_ENV_E, fs=8)
    lbl(ax, 9.85, 3.66, t['forward'], fs=8)
    # #3: forwarded task -> neighbour's queue, short top-right cue (no full-width sweep)
    arrow(ax, (10.83, 5.77), (11.25, 6.55), color=TASK_FLOW, ls=(0, (5, 2)), lw=1.3, rad=-0.25)
    lbl(ax, 10.78, 6.78, t['to_neighbor'], fs=7.4, color='#555', style='italic')
    # #2: DVFS closed-form moved beside the DVFS CPU box, tied with a leader line
    ax.plot([8.62, 9.4], [1.92, 2.78], color='#9aa6b6', lw=1.0, ls=(0, (2, 2)), zorder=2)
    lbl(ax, 10.0, 3.5, t['dvfs_head'], fs=7.6, color='#333', weight='bold')
    lbl(ax, 10.0, 2.98, t['dvfs_eq'], fs=7.2, color='#333', bg='#f7f9fc')

    # ---- region (c): battery energy cycle ----
    box(ax, 12.05, 1.05, 3.65, 7.15, fc='#fffdf8', ec='#e6cfa6', lw=1.1)
    lbl(ax, 13.87, 7.95, t['c_label'], fs=11, weight='bold')
    box(ax, 12.45, 6.6, 2.9, 0.95, t['solar'], fc='#fff1c9', ec='#e0a800', fs=9)
    box(ax, 12.45, 4.95, 2.9, 1.0, t['battery'], fc='#eaf2ea', ec=C_CRI_E, fs=9.5, bold=True)
    box(ax, 12.45, 3.2, 2.9, 1.0, t['discharge'], fc='#fde7e7', ec=C_ACT_E, fs=8.6)
    box(ax, 12.45, 1.35, 2.9, 1.15, t['health'], fc='#f6d2d2', ec=C_ACT_E, fs=9, bold=False)
    arrow(ax, (13.9, 6.6), (13.9, 5.95), color=ENE_FLOW, ls=(0, (4, 2)))
    arrow(ax, (13.9, 3.2), (13.9, 4.0), color=ENE_FLOW, ls=(0, (4, 2)))   # discharge into battery (energy out)
    arrow(ax, (13.9, 4.95), (13.9, 4.22), color=TASK_FLOW)
    arrow(ax, (13.9, 3.2), (13.9, 2.5))

    # ---- legend ----
    lx = 0.6
    for dx, (col, ls, txt) in enumerate([
            (TASK_FLOW, '-', t['lg_task']),
            (ENE_FLOW, (0, (4, 2)), t['lg_ene']),
            (INFO_FLOW, (0, (1, 2)), t['lg_info'])]):
        x0 = lx + dx * 4.7
        ax.plot([x0, x0 + 0.7], [0.55, 0.55], color=col, lw=2.0, ls=ls)
        lbl(ax, x0 + 0.85, 0.55, txt, fs=9, ha='left')
    # info broadcast example (sat -> neighbours, dotted)
    arrow(ax, (4.5, 6.7), (5.65, 6.0), color=INFO_FLOW, ls=(0, (1, 2)), lw=1.2, style='-|>')

    plt.tight_layout()
    fname = 'arch_system_model.png' if lang == 'en' else f'arch_system_model_{lang}.png'
    p = os.path.join(OUT, fname)
    plt.savefig(p, dpi=160, bbox_inches='tight'); plt.close()
    return p


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — LyaMAPPO framework (CTDE)
# ══════════════════════════════════════════════════════════════════════
def fig_framework():
    set_lang_font('en')  # framework labels are English; never inherit zh font
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    ax.text(8, 8.66, 'LyaMAPPO Framework — Lyapunov Drift-plus-Penalty $\\otimes$ MAPPO (CTDE)',
            ha='center', fontsize=15, fontweight='bold')

    # ---- CTDE bands ----
    ax.add_patch(FancyBboxPatch((0.25, 4.55), 15.5, 3.55, boxstyle='round,pad=0,rounding_size=0.12',
                                fc='#f3f7fc', ec='#aebfd6', lw=1.2, zorder=0))
    ax.add_patch(FancyBboxPatch((0.25, 0.5, ), 15.5, 3.7, boxstyle='round,pad=0,rounding_size=0.12',
                                fc='#f2f8f2', ec='#aed2ae', lw=1.2, zorder=0))
    lbl(ax, 0.55, 7.86, 'Decentralized execution / rollout  (per satellite, local 54-d obs)',
        fs=10.5, ha='left', weight='bold', color='#3f6090')
    lbl(ax, 0.55, 3.92, 'Centralized training  (offline, global info)',
        fs=10.5, ha='left', weight='bold', color='#2e7d32')

    # ---- top band: Environment, Actor, reward ----
    box(ax, 0.5, 5.0, 2.75, 2.7, fc=C_ENV, ec=C_ENV_E, lw=1.6)
    lbl(ax, 1.87, 7.45, 'Environment', fs=10.5, weight='bold')
    lbl(ax, 1.87, 6.35, 'N satellites\nQueues:\n$Q^F_n$ forward\n$Q^B_n$ compute\n$z_n$ DoD virtual (Neely)\n\nshared DVFS / battery / ISL',
        fs=8.6)
    star(ax, 3.0, 6.05, 2); star(ax, 0.78, 5.2, 6)

    box(ax, 4.05, 5.05, 4.5, 2.6, fc=C_ACT, ec=C_ACT_E, lw=1.7)
    lbl(ax, 6.3, 7.42, 'Actor  $\\pi_\\theta$  (decentralized)', fs=10.5, weight='bold')
    lbl(ax, 6.3, 6.25, 'input 54-d / task:\n[ ID 1 + local 10 (incl. $z_n$) + neighbor 9$\\times$4 + task 7 ]\n'
                       'LayerNorm $\\to$ 256 $\\to$ 256 $\\to$ 5 $\\to$ masked-softmax $\\to$ sample\n'
                       'sequential per-task decision (act_one)', fs=8.5)
    star(ax, 8.3, 5.25, 5)

    box(ax, 9.35, 5.0, 6.35, 2.7, fc=C_LYA, ec=C_LYA_E, lw=1.7)
    lbl(ax, 12.52, 7.46, 'Lyapunov reward construction', fs=10.5, weight='bold')
    lbl(ax, 12.52, 6.74, '$L=\\frac{1}{2}\\sum_n[(Q^F_n)^2+(Q^B_n)^2+\\eta z_n^2]$,  $\\eta{=}0.5$', fs=9.2)
    lbl(ax, 12.52, 6.16, 'drift-plus-penalty:  $\\Delta L + V\\cdot C_n$,   $V{=}50$', fs=9.2)
    lbl(ax, 12.52, 5.42, 'outcome-aware:  $R_n=r_n+w_d\\,$done$-w_t\\,$to$-w_r\\,$rej$-w_h\\,$HL$-w_q Q$\n'
                         'weights  $(w_d,w_t,w_r,w_h,w_q)=(10,5,5,2,0.05)$', fs=8.5)
    star(ax, 9.62, 6.16, 1); star(ax, 9.62, 5.42, 3)

    # ---- bottom band: Buffer, Critic, PPO ----
    box(ax, 0.5, 1.0, 2.75, 2.55, 'Rollout buffer\nstore $(s_n, a_n, R_n)$\nper-slot + per-task', fc='#eef2f7', ec=C_QUE_E, fs=9)
    box(ax, 4.05, 0.95, 4.5, 2.7, fc=C_CRI, ec=C_CRI_E, lw=1.7)
    lbl(ax, 6.3, 3.4, 'Critic  $V_\\phi$  (centralized)', fs=10.5, weight='bold')
    lbl(ax, 6.3, 2.25, 'input 245-d = 5 nodes$\\times$47 + global summary 10\n'
                       '$\\Rightarrow$ decoupled from N (scalable)\n'
                       'LayerNorm $\\to$ 256 $\\to$ 256 $\\to$ 1', fs=8.7)
    star(ax, 8.3, 3.4, 4)

    box(ax, 9.35, 0.95, 6.35, 2.7, fc='#ede7f6', ec='#7e57c2', lw=1.6)
    lbl(ax, 12.52, 3.4, 'PPO update', fs=10.5, weight='bold')
    lbl(ax, 12.52, 2.45, 'GAE  ($\\gamma{=}0.99$, $\\lambda{=}0.95$)\n'
                         'task-level advantage:  $A_{task}=A_{slot}+\\beta\\frac{r-\\bar r}{\\sigma}$,  $\\beta{=}0.5$\n'
                         'clipped PPO  ($\\epsilon{=}0.2$, entropy $\\beta_e{=}0.02$, epoch 2)', fs=8.6)
    star(ax, 15.4, 2.95, 7)

    # ---- flows ----
    # execution: env -> actor (obs), actor -> env (action, curved over top)
    arrow(ax, (3.25, 6.6), (4.05, 6.6)); lbl(ax, 3.65, 6.92, 'obs $s_n$\n54-d', fs=7.8)
    arrow(ax, (6.3, 7.65), (1.87, 7.7), color=TASK_FLOW, rad=-0.32, lw=1.5)
    lbl(ax, 4.0, 8.18, 'action $a\\in\\{0..4\\}\\;\\to\\;$ env evolves (DVFS / battery / queues)', fs=8.2, color='#333')
    # env/actor transitions -> reward
    arrow(ax, (8.55, 6.3), (9.35, 6.3)); lbl(ax, 8.95, 6.62, 'outcomes', fs=7.6)
    # reward -> buffer (cross band, down-left)
    arrow(ax, (12.0, 5.0), (3.0, 3.55), color=TASK_FLOW, rad=0.12, lw=1.5)
    lbl(ax, 7.4, 4.35, 'store transitions $(s_n,a_n,R_n)$', fs=8, color='#333', bg='#ffffff')
    # env -> critic (global state, training only)
    arrow(ax, (1.6, 5.0), (1.6, 3.55), color=INFO_FLOW, lw=1.5)
    lbl(ax, 0.95, 4.3, 'global\n245-d', fs=7.6, color='#555', ha='center')
    # buffer -> critic -> ppo
    arrow(ax, (3.25, 2.2), (4.05, 2.2)); lbl(ax, 3.65, 2.5, 'minibatch', fs=7.4)
    arrow(ax, (8.55, 2.2), (9.35, 2.2)); lbl(ax, 8.95, 2.5, '$V(s)$,\nGAE', fs=7.6)
    # feedback: ppo -> actor & critic (dashed updates)
    arrow(ax, (12.52, 3.65), (6.3, 5.05), color='#7e57c2', ls=(0, (5, 2)), lw=1.5, rad=-0.16)
    lbl(ax, 10.4, 4.5, 'update $\\theta$ (actor)', fs=7.8, color='#5e35b1', bg='#ffffff')
    arrow(ax, (9.35, 2.05), (8.55, 2.05), color='#7e57c2', ls=(0, (5, 2)), lw=1.5)
    lbl(ax, 8.0, 1.55, 'update $\\phi$ (critic)', fs=7.6, color='#5e35b1')

    # innovation legend
    inno = ('★ innovations:  1 Lyapunov$\\otimes$MAPPO coupling   2 $z_n$ in actor state   '
            '3 outcome-aware reward   4 N-decoupled critic   5 sequential decision   '
            '6 shared DVFS base   7 task-level advantage')
    ax.text(8, 0.16, inno, ha='center', va='center', fontsize=8.0, color=STAR, fontweight='bold')

    plt.tight_layout()
    p = os.path.join(OUT, 'arch_lyamappo_framework.png')
    plt.savefig(p, dpi=160, bbox_inches='tight'); plt.close()
    return p


if __name__ == '__main__':
    import sys
    langs = sys.argv[1:] if len(sys.argv) > 1 else ['en', 'zh']
    written = [fig_system_model(lang=lg) for lg in langs]
    written.append(fig_framework())
    print('Architecture figures written:')
    for w in written:
        print(' -', w)
