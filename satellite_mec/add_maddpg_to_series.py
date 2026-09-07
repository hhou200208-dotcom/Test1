"""增量式:只评估 MADDPG_DoD(LyDRL-DoD),合并进现有 series/scoreboard JSON。

复刻 eval_series.py 的 cap() 口径(同 seed=get_eval_seeds(0)、同 5400 槽、默认
lyapunov_calc),只跑一个策略并 merge —— 其它 7 策略的曲线保持已提交数据不动,
零漂移。产物供 plot_pdf_figures.py / plot_paper_figures.py 出图1-5。
"""
import json
import numpy as np
from core import Config, SatelliteMECEnv
from baselines.maddpg_dod import MADDPGDoDPolicy

SERIES = 'docs/series_lh4_n25.json'
SCALAR = 'docs/scoreboard7_lh4.json'


class _C(Config):
    LAMBDA_HIGH = 4.0
    LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)


cfg = _C(); cfg.N_EVAL_RUNS = 1
env = SatelliteMECEnv(cfg)                      # 默认 lyapunov_calc,与 eval_series 一致

series = json.load(open(SERIES))
scalar = json.load(open(SCALAR))
print('existing policies:', list(series.keys()))

md = MADDPGDoDPolicy(cfg, env); md.load('checkpoints/MADDPG_DoD_lh4_32K')

# ── 复刻 eval_series.cap() —— 只跑 MADDPG_DoD ──
env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); md.set_eval_mode()
E = []; D = []; H = []; Q = []; SAT = []; DEN = []; dod = []; qmb = []; alld = []
for _ in range(cfg.T_EVAL):
    _, _, _, info = env.step(policy=md)
    E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
    H.append(info['avg_health_loss']);    Q.append(info['queue_task_count'])
    SAT.append(info['slot_satisfaction_rate']); DEN.append(info['done_tasks'] + info['slot_timeout'])
    dod.append(info['avg_dod']);          qmb.append(info['total_queue_size'] / 1e6)
    alld.extend(info['slot_e2e_delays'])
sat = float(info.get('eval_satisfaction_rate', float('nan')))

series['MADDPG_DoD'] = {'satisfaction': sat, 'energy': E, 'delay': D, 'hl': H,
                        'queue_tasks': Q, 'sat_slot': SAT, 'sat_denom': DEN}
scalar['MADDPG_DoD'] = {'satisfaction': sat,
                        'delay': float(np.mean(alld)) if alld else 0.0,
                        'hl': float(np.mean(H)), 'dod': float(np.mean(dod)),
                        'queue_mb': float(np.mean(qmb)), 'queue_tasks': float(np.mean(Q)),
                        'cr_internal': float(env.get_eval_completion_rate())}

json.dump(series, open(SERIES, 'w'))
json.dump(scalar, open(SCALAR, 'w'), indent=2)
print(f'MERGED MADDPG_DoD -> Sat={sat:.4f} cumHL={np.sum(H):.3e} '
      f'meanDelay(sum/slot)={np.mean(D):.1f} meanQ={np.mean(Q):.2f} '
      f'ΣE={np.sum(E)/1e3:.0f}kJ')
print('policies now:', list(series.keys()))
