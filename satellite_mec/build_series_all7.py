"""Build combined per-slot series for all 7 methods (new params, 1 seed) ->
docs/pack_all7/series_all7.json. Reuses the 5 cached methods; evals the 2
ablation checkpoints (w/o DoD, w/o Task Prior).

New physics: kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD[0,0.8].
Each entry: {E: slot_system_energy[J], D: slot_total_delay[s], hl: avg_health_loss,
             sat_slot: slot_satisfaction_rate, sat_denom: done+timeout}.
"""
import os, json, time
import numpy as np
from core import Config, SatelliteMECEnv
from training import MAPPOPolicy

class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0

cfg = _C(); cfg.N_EVAL_RUNS = 1; cfg.V_DVFS = 2e17
env = SatelliteMECEnv(cfg)
T = cfg.T_EVAL
OUT = 'docs/pack_all7'; os.makedirs(OUT, exist_ok=True)

# 5 cached methods (map cache keys -> display names)
cache = json.load(open('docs/newparams_series/series.json'))
KEYMAP = {'LyaMAPPO': 'BLA-MAPPO', 'MADDPG_DoD': 'LyDRL-DoD',
          'MHSPO': 'MHSPO', 'GDCO': 'GDCO', 'LocalOnly': 'LSO'}
data = {}
for k, disp in KEYMAP.items():
    s = cache[k]
    data[disp] = {kk: s[kk] for kk in ('E', 'D', 'hl', 'sat_slot', 'sat_denom')}


def eval_ckpt(path):
    p = MAPPOPolicy(cfg, name='e'); p.load(path)
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); p.set_eval_mode()
    E = []; D = []; H = []; SAT = []; DEN = []
    for _ in range(T):
        _, _, _, info = env.step(policy=p)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
        H.append(info['avg_health_loss']); SAT.append(info['slot_satisfaction_rate'])
        DEN.append(info['done_tasks'] + info['slot_timeout'])
    return {'E': E, 'D': D, 'hl': H, 'sat_slot': SAT, 'sat_denom': DEN}


for disp, ck in [('w/o DoD', 'checkpoints/MAPPO_NoBat_lh4_8K'),
                 ('w/o Task Prior', 'checkpoints/wo_task_prior_16k/step_12000')]:
    t0 = time.time(); data[disp] = eval_ckpt(ck)
    print(f'{disp}: cumHL={np.cumsum(data[disp]["hl"])[-1]*192:.2f} [{time.time()-t0:.0f}s]', flush=True)

json.dump(data, open(os.path.join(OUT, 'series_all7.json'), 'w'))
print('[done] wrote', os.path.join(OUT, 'series_all7.json'), '| methods:', list(data.keys()), flush=True)
