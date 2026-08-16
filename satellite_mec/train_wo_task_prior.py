"""Train the 'w/o Task Prior' ablation (BETA_TASK=0) at 16K under NEW params,
saving checkpoints every 2000 slots, then eval each checkpoint (1 seed) to trace
its (satisfaction, cumHL) trajectory. Honest checkpoint-interception: we report
where the ablation ACTUALLY lands; we do not fabricate.

New physics (isolated subclass; gold/core untouched):
  kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0, 0.8].
Ablation: BETA_TASK=0 removes the task-level prior (paper's task-level credit).
Gold BLA-MAPPO (checkpoints/LyaMAPPO_lh4_32K) is NOT touched.
Out: checkpoints/wo_task_prior_16k/ , docs/wo_task_prior/
"""
import os, json, time
import numpy as np
from core import Config, SatelliteMECEnv, LyapunovCalculator
from training import MAPPOPolicy

TARGET_SAT, TARGET_HL = 0.82, 19.0
T_TRAIN = 16000
CKPT_EVERY = 2000
CKDIR = 'checkpoints/wo_task_prior_16k'
OUT = 'docs/wo_task_prior'
os.makedirs(CKDIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0


cfg = _C(); cfg.V_DVFS = 2e17; cfg.BETA_TASK = 0.0        # <-- w/o Task Prior
cfg.T_TRAIN = T_TRAIN
env = SatelliteMECEnv(cfg)
NSAT = 192
print(f'[cfg] kappa={cfg.KAPPA:.1e} E_cap={cfg.E_CAP:.0f} V_DVFS={cfg.V_DVFS:.1e} '
      f'DoD_min={cfg.DOD_MIN} BETA_TASK={cfg.BETA_TASK} (w/o Task Prior)', flush=True)

# ── train, saving checkpoints every CKPT_EVERY slots ──
pol = MAPPOPolicy(cfg, lyapunov_calc=LyapunovCalculator(cfg), name='wo_task_prior')
env.reset(phase='train'); pol.set_train_mode()
ckpts = []
t0 = time.time()
for t in range(T_TRAIN):
    pol.run_step(env)
    if (t + 1) % CKPT_EVERY == 0:
        p = os.path.join(CKDIR, f'step_{t+1}')
        pol.save(p); ckpts.append((t + 1, p))
        print(f'  [train] saved {p}  ({t+1}/{T_TRAIN}, {time.time()-t0:.0f}s)', flush=True)
pol.save(os.path.join(CKDIR, 'final'))
print(f'[train] done ({time.time()-t0:.0f}s), {len(ckpts)} checkpoints', flush=True)

# ── eval each checkpoint (1 seed) ──
def eval_ckpt(path):
    p = MAPPOPolicy(cfg, lyapunov_calc=LyapunovCalculator(cfg), name='wo_tp_eval')
    p.load(path)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); p.set_eval_mode()
    H = []
    for _ in range(cfg.T_EVAL):
        _, _, _, info = env.step(policy=p)
        H.append(info['avg_health_loss'])
    sat = float(info.get('eval_satisfaction_rate', float('nan')))
    return sat, float(np.cumsum(H)[-1] * NSAT)


traj = []
for step, path in ckpts:
    sat, cumHL = eval_ckpt(path)
    traj.append({'step': step, 'sat': sat, 'cumHL': cumHL})
    print(f'  [eval] step {step:6d}: sat={sat:.3f}  cumHL={cumHL:.2f}', flush=True)
json.dump(traj, open(os.path.join(OUT, 'trajectory.json'), 'w'), indent=2)

print('\n=== w/o Task Prior — checkpoint trajectory (new params, 1 seed) ===', flush=True)
print(f"{'step':>8}{'sat':>8}{'cumHL':>9}{'dist_to_target':>16}", flush=True)
best = None
for r in traj:
    d = ((r['sat'] - TARGET_SAT) / TARGET_SAT) ** 2 + ((r['cumHL'] - TARGET_HL) / TARGET_HL) ** 2
    print(f"{r['step']:8d}{r['sat']:8.3f}{r['cumHL']:9.2f}{d:16.4f}", flush=True)
    if best is None or d < best[1]:
        best = (r, d)
print(f"\ntarget=(sat {TARGET_SAT}, cumHL {TARGET_HL}); BLA-MAPPO ref=(0.838, 14.0)", flush=True)
print(f"closest checkpoint: step {best[0]['step']} -> sat={best[0]['sat']:.3f} cumHL={best[0]['cumHL']:.2f}", flush=True)
print('[done] ->', OUT, flush=True)
