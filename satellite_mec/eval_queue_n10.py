"""n=10 multi-seed system queue-length-over-time comparison (λ_high=4, N=25→192).

For each method: eval 10 seeds × 5400 slots, record per-slot queue_task_count,
average across seeds (mean ± std). Plot system queue length (per-sat mean × 192,
150-slot smoothed) vs time slot, with ±std shaded band.
Resumable: per-method result cached to docs/queue_n10_data.json.
Output figure: docs/figures_queue_n10/queue_backlog_n10.png
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import Config, SatelliteMECEnv
from baselines import LocalOnlyPolicy, MHSPOPolicy, GDCOPolicy
from baselines.maddpg_dod import MADDPGDoDPolicy
from training import MAPPOPolicy


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)


cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)
NSEED = 10; NSAT = 192
OUT = 'docs/figures_queue_n10'; os.makedirs(OUT, exist_ok=True)
DATA = 'docs/queue_n10_data.json'
data = json.load(open(DATA)) if os.path.exists(DATA) else {}


def eval_seeds(pol, label, warmup=False):
    if label in data:
        print(f'{label}: cached, skip', flush=True); return
    if warmup:
        env.reset(phase='train')
        for _ in range(cfg.T_WARMUP):
            env.step(policy=pol)
    Qs = []
    for i in range(NSEED):
        env.reset(phase='eval', seeds=cfg.get_eval_seeds(i)); pol.set_eval_mode()
        Q = []; t0 = time.time()
        for _ in range(cfg.T_EVAL):
            _, _, _, info = env.step(policy=pol)
            Q.append(info['queue_task_count'])
        Qs.append(Q)
        print(f'{label} seed{i}: Qmean={np.mean(Q):.2f} [{time.time()-t0:.0f}s]', flush=True)
    Qs = np.asarray(Qs)                                   # (10, 5400)
    data[label] = {'q_mean': Qs.mean(0).tolist(), 'q_std': Qs.std(0).tolist()}
    json.dump(data, open(DATA, 'w'))
    print(f'== {label} DONE (n={NSEED}), saved ==', flush=True)


eval_seeds(LocalOnlyPolicy(cfg, env), 'LocalOnly')
eval_seeds(GDCOPolicy(cfg, env), 'GDCO')
eval_seeds(MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0), 'MHSPO', warmup=True)
md = MADDPGDoDPolicy(cfg, env); md.load('checkpoints/MADDPG_DoD_lh4_32K')
eval_seeds(md, 'MADDPG_DoD')
lya = MAPPOPolicy(cfg, name='LyaMAPPO'); lya.load('checkpoints/LyaMAPPO_lh4_32K')
eval_seeds(lya, 'LyaMAPPO')

# ── plot ──
ORDER = ['LyaMAPPO', 'MADDPG_DoD', 'MHSPO', 'GDCO', 'LocalOnly']
DISP = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD', 'LocalOnly': 'LSO'}
COLOR = {'LyaMAPPO': '#d62728', 'MADDPG_DoD': '#17becf', 'MHSPO': '#2ca02c',
         'GDCO': '#9467bd', 'LocalOnly': '#7f7f7f'}


def smooth(a, w=150):
    a = np.asarray(a, float); return np.convolve(a, np.ones(w) / w, 'valid')


fig, ax = plt.subplots(figsize=(8, 5))
for p in ORDER:
    if p not in data:
        continue
    m = smooth(np.asarray(data[p]['q_mean']) * NSAT)
    s = smooth(np.asarray(data[p]['q_std']) * NSAT)
    x = np.arange(len(m)); big = (p == 'LyaMAPPO')
    ax.plot(x, m, color=COLOR[p], lw=2.4 if big else 1.5, label=DISP.get(p, p),
            zorder=3 if big else 2)
    ax.fill_between(x, m - s, m + s, color=COLOR[p], alpha=0.12, lw=0)
ax.set_xlabel('Time slot'); ax.set_ylabel('System queue length')
ax.legend(fontsize=9, ncol=2); ax.grid(True, ls=':', alpha=0.4)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'queue_backlog_n10.png'), dpi=180); plt.close()
print('figure written ->', os.path.join(OUT, 'queue_backlog_n10.png'), flush=True)
