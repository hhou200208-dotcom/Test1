# -*- coding: utf-8 -*-
"""
plot_8methods_kde.py —— 8 方法对比图(读两个 JSON 合并,详细注释版)

数据来源(两个独立 JSON,schema 相同):
    JSON_MAIN = docs/pack_all7/series_all7.json          你原有的 7 方法逐时隙序列
                {BLA-MAPPO, w/o DoD, w/o Task Prior, MHSPO, LyDRL-DoD, GDCO, LSO}
    JSON_LD   = docs/pack_linear_dod/series_linear_dod.json  本次跑的消融
                只从这里取 'w/ Linear-DoD' 一条,其余(如 BLA-MAPPO)忽略
    合并后 = 7 + 1 = 8 方法,一起画。

输出 4 张图(与 plot_4figs_all7 同风格):
    m8_fig1_satisfaction / m8_fig2_delay / m8_fig3_energy / m8_fig4_cumulative_hl

两个 JSON 的每个方法都是 {'E','D','hl','sat_slot','sat_denom'},逐时隙 list(长 5400)。
schema 一致所以能直接合并;若某文件缺失会报错提示,而不是静默出错。

【量纲约定】E、D 外延量(25星求和)-> x(192/25)=RATIO;sat_slot 比率 -> 不投影;
hl 每星平均 -> x192=NSAT(不是 xRATIO)。RATIO≠NSAT,勿混用。
依赖:numpy / scipy / matplotlib   运行:python plot_8methods_kde.py
"""
import os, json
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')                       # 先切后端:纯文件输出,无需 GUI
import matplotlib.pyplot as plt

# ======================================================================
# 0. 路径、常数、两个 JSON 的合并
# ======================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
JSON_MAIN = os.path.join(HERE, 'docs', 'pack_all7', 'series_all7.json')          # 7 方法
JSON_LD   = os.path.join(HERE, 'docs', 'pack_linear_dod', 'series_ld_1seed.json')  # 取 w/ Linear-DoD(重跑逐槽)
OUT = os.path.join(HERE, 'docs', 'pack_all7'); os.makedirs(OUT, exist_ok=True)   # 输出目录(可改)

RATIO = 192.0 / 25.0     # =7.68,外延量 25星->192星
NSAT  = 192              # 健康损耗:每星平均 x192 = 全系统累计

LD_KEY = 'w/ Linear-DoD'                 # ← 只从 JSON_LD 取这一条(改这里可换别的键)

# ---- 载入并合并 ----
for p in (JSON_MAIN, JSON_LD):           # 缺文件时给明确报错,而非后面 KeyError
    if not os.path.exists(p):
        raise FileNotFoundError(f'找不到数据文件: {p}')
data = json.load(open(JSON_MAIN))                        # 先装 7 方法
ld = json.load(open(JSON_LD))
if LD_KEY not in ld:
    raise KeyError(f"{JSON_LD} 里没有 '{LD_KEY}';现有键: {list(ld.keys())}")
data[LD_KEY] = ld[LD_KEY]                                # 只把 w/ Linear-DoD 并进来(其余忽略)

# ======================================================================
# 1. 样式配置(8 方法)
# ======================================================================
# 绘制/图例顺序:主方法 BLA-MAPPO 打头;3 个消融(w/o DoD、w/o Task Prior、w/ Linear-DoD)聚在一起;
# 4 个基线殿后。ORDER 里写谁画谁,JSON 没有的自动过滤。
ORDER = ['BLA-MAPPO', 'w/o DoD', 'w/o Task Prior', 'w/ Linear-DoD',
         'MHSPO', 'LyDRL-DoD', 'GDCO', 'LSO']
ORDER = [m for m in ORDER if m in data]

# 配色:前 7 个沿用你 plot_4figs_all7 的配色;w/ Linear-DoD 用蓝色(#1f77b4),与其余都不撞。
COLOR = {'BLA-MAPPO': '#d62728', 'w/o DoD': '#e377c2', 'w/o Task Prior': '#8c564b',
         'w/ Linear-DoD': '#1f77b4',
         'MHSPO': '#2ca02c', 'LyDRL-DoD': '#F2C200', 'GDCO': '#9467bd', 'LSO': '#7f7f7f'}
# 标记形状:保证黑白/色盲可分;w/ Linear-DoD 用倒三角 'v'(与已有 P/o/s/^/D/X/* 都不同)。
MARK = {'BLA-MAPPO': 'P', 'w/o DoD': 'o', 'w/o Task Prior': 's', 'w/ Linear-DoD': 'v',
        'MHSPO': '^', 'LyDRL-DoD': 'D', 'GDCO': 'X', 'LSO': '*'}
# 显示名映射:key=JSON 原始名(不可改),value=图例显示名;没列出的原样显示。
DISPLAY = {'BLA-MAPPO': 'BLA-MAPPO', 'w/o DoD': 'w/o DoD', 'w/o Task Prior': 'w/o Task Prior',
           'w/ Linear-DoD': 'w/ Linear-DoD', 'MHSPO': 'MHSPO', 'LyDRL-DoD': 'LyDRL-DoD',
           'GDCO': 'GDCO', 'LSO': 'LSO'}
disp = lambda k: DISPLAY.get(k, k)
big  = lambda k: k == 'BLA-MAPPO'        # 主方法:线宽/标记/zorder 加强
T = len(data[ORDER[0]]['hl'])

# ======================================================================
# 2. 通用 KDE 分布图函数(fig1~3 共用)
# ======================================================================
def kde_fig(vals, xlabel, fname, smooth=2.2):
    # 2.1 统一 x 轴:0.3%/99.7% 分位数定界剔离群点;isfinite 去 NaN/inf
    allv = np.concatenate([np.asarray(v, float)[np.isfinite(np.asarray(v, float))]
                           for v in vals.values()])
    lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
    xs = np.linspace(lo, hi, 400)                        # 评估网格 400 点
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for k in ORDER:
        v = np.asarray(vals[k], float); v = v[np.isfinite(v)]
        if len(v) < 10 or v.std() < 1e-9:                # 2.2 退化跳过(否则 KDE 协方差奇异)
            continue
        kde = gaussian_kde(v); kde.set_bandwidth(kde.factor * smooth)   # 2.3 带宽 x smooth
        ys = kde(xs)
        ax.plot(xs, ys, color=COLOR[k], lw=2.6 if big(k) else 1.6,      # 2.4 密度曲线
                zorder=3 if big(k) else 2)
        mk = np.linspace(0, len(xs) - 1, 16).astype(int)               # 2.5 稀疏 16 标记
        ax.plot(xs[mk], ys[mk], color=COLOR[k], marker=MARK[k], ls='none',
                markersize=6 if big(k) else 5, label=disp(k), zorder=3 if big(k) else 2)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)           # 2.6 密度非负
    ax.set_xlabel(xlabel); ax.set_ylabel('Probability density')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)     # 8 条目分 2 列
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'pdf'): plt.savefig(os.path.join(OUT, f'{fname}.{ext}'), dpi=180)
    plt.close()

# ======================================================================
# 3. Fig1 满意度(比率,不投影;只留 sat_denom>0 的时隙)
# ======================================================================
sat = {}
for k in ORDER:
    s = np.asarray(data[k]['sat_slot'], float); d = np.asarray(data[k]['sat_denom'], float)
    sat[k] = s[d > 0]
kde_fig(sat, 'User satisfaction', 'm8_fig1_satisfaction')

# ======================================================================
# 4. Fig2 系统总时延(外延量 xRATIO,单位 s)
# ======================================================================
kde_fig({k: np.asarray(data[k]['D']) * RATIO for k in ORDER}, 'System total delay (s)', 'm8_fig2_delay')

# ======================================================================
# 5. Fig3 系统总能耗(外延量 xRATIO,再 /1e3 J->kJ)
# ======================================================================
kde_fig({k: np.asarray(data[k]['E']) * RATIO for k in ORDER}, 'System total energy (J)', 'm8_fig3_energy')

# ======================================================================
# 6. Fig4 累计健康损耗时序(hl 每星平均 x192;斜率=速率,终点=累计量)
# ======================================================================
fig, ax = plt.subplots(figsize=(7.2, 5))
for k in ORDER:
    y = np.cumsum(np.asarray(data[k]['hl'], float)) * NSAT
    ax.plot(np.arange(T), y, color=COLOR[k], lw=2.6 if big(k) else 1.6,
            label=disp(k), zorder=3 if big(k) else 2)
ax.set_xlabel('Time slot'); ax.set_ylabel('System cumulative health loss')
ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf'): plt.savefig(os.path.join(OUT, f'm8_fig4_cumulative_hl.{ext}'), dpi=180)
plt.close()

print('[done] 8 方法图已写入', OUT)
print('  方法顺序:', ORDER)
