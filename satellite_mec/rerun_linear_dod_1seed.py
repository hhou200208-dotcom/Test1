"""w/ Linear-DoD 单 seed 重跑(容器回收后重建逐槽 series)。
口径与 build_series_all7.py 完全一致:新物理 κ=1.5e-27, E_cap=54kJ, V_DVFS=2e17, DoD[0,0.8],
λ=4, T_EVAL=5400, 单 seed。训练用线性老化(LINEAR_DOD_LOSS=True), 评估用真凸测量(=False)。
逐槽记 {E,D,hl,sat_slot,sat_denom}(与 series_all7 同 schema),写:
  docs/pack_linear_dod/series_ld_1seed.json   (只含 'w/ Linear-DoD')
并打印标量,供与 5-seed std 比对(sat 0.772±0.026, delay 386.3±18.0, energy 0.242±0.012, cumHL 11.64±1.89)。
用法: python rerun_linear_dod_1seed.py --t_train 8000
"""
import argparse, os, json, time
import numpy as np
from core import Config, SatelliteMECEnv, LyapunovCalculator
from evaluation import ExperimentRunner
from training import MAPPOPolicy

RATIO = 192 / 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t_train', type=int, default=8000)
    args = ap.parse_args()

    class _C(Config):                         # 与 build_series_all7.py 同口径
        LAMBDA_HIGH = 4.0
        LAMBDA = 4.0 * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
        KAPPA = 1.5e-27
        E_CAP = 54_000.0
        DOD_MIN = 0.0
    cfg = _C(); cfg.N_EVAL_RUNS = 1; cfg.V_DVFS = 2e17; cfg.T_TRAIN = args.t_train

    # ── 训练:线性老化 ──
    cfg.LINEAR_DOD_LOSS = True
    env = SatelliteMECEnv(cfg)
    runner = ExperimentRunner(cfg, 'LinearDoD_1seed_8K', debug=False)
    runner.setup_algorithm_dir('MAPPO')
    lyap = LyapunovCalculator(cfg, linear_loss=True)
    env.lyapunov_calc = lyap
    mappo = MAPPOPolicy(cfg, lyapunov_calc=lyap, name='MAPPO')
    print(f'[LinearDoD 1seed] 训练 {args.t_train} 槽, LINEAR_DOD_LOSS=True, 新物理 V_DVFS={cfg.V_DVFS:.1e}', flush=True)
    t0 = time.time()
    runner.run_training(mappo, env)
    mappo.save('checkpoints/linear_dod_1seed_8k',
               extra_info={'variant': 'linear', 't_train': args.t_train, 'lambda_high': 4.0})

    # ── 评估:真凸测量,逐槽记 series_all7 口径 ──
    cfg.LINEAR_DOD_LOSS = False
    env.reset(phase='eval', seeds=cfg.get_eval_seeds(0)); mappo.set_eval_mode()
    E, D, H, S, N = [], [], [], [], []
    for _ in range(cfg.T_EVAL):
        _, _, _, info = env.step(policy=mappo)
        E.append(info['slot_system_energy']); D.append(sum(info['slot_e2e_delays']))
        H.append(info['avg_health_loss']); S.append(info['slot_satisfaction_rate'])
        N.append(info['done_tasks'] + info['slot_timeout'])
    series = {'E': E, 'D': D, 'hl': H, 'sat_slot': S, 'sat_denom': N}

    os.makedirs('docs/pack_linear_dod', exist_ok=True)
    json.dump({'w/ Linear-DoD': series}, open('docs/pack_linear_dod/series_ld_1seed.json', 'w'))

    # ── 标量 + 与 5-seed std 比对 ──
    Sa = np.asarray(S); Nn = np.asarray(N)
    sat = float(Sa[Nn > 0].mean())
    delay = float((np.asarray(D) * RATIO).mean())
    energy = float((np.asarray(E) * RATIO / 1e3).mean())
    cumhl = float(np.sum(H) * 192)
    ref = {'satisfaction': (0.772, 0.026), 'delay': (386.3, 18.0),
           'energy': (0.242, 0.012), 'cumHL': (11.64, 1.89)}
    got = {'satisfaction': sat, 'delay': delay, 'energy': energy, 'cumHL': cumhl}
    print(f'\n[LinearDoD 1seed] DONE [{time.time()-t0:.0f}s]  逐槽 series 已存', flush=True)
    print(f"{'指标':<14}{'本次':>10}{'5seed均值±std':>18}{'在±1std内?':>12}", flush=True)
    for k in ['satisfaction', 'delay', 'energy', 'cumHL']:
        m, s = ref[k]; v = got[k]; inside = abs(v - m) <= s
        print(f"{k:<14}{v:>10.3f}{f'{m}±{s}':>18}{'✓' if inside else '✗ 超出':>12}", flush=True)


if __name__ == '__main__':
    main()
