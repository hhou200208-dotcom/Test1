"""sensitivity_v_paper.py
==========================
V sensitivity analysis at λ_high = 4.0 (main paper scenario).

For each V in {5, 25, 50, 100, 250}:
  - Train LyaMAPPO 8K steps with paper-final hyperparameters
    (BETA=0.02, W_DONE=10, W_HL=2, W_TIMEOUT=5, W_REJECT=5)
    while overriding the Lyapunov V trade-off knob
  - Evaluate ONLY MAPPO for T_EVAL slots
  - Record CR, Sat, e2e_delay, HL, DoD, Queue

Baseline values from the canonical final_b run are loaded for reference.

Usage:
    python sensitivity_v_paper.py [--t_train N] [--v_grid 5,25,50,...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv, LyapunovCalculator
from evaluation import MetricsRecorder, ExperimentRunner
from training import MAPPOPolicy


V_GRID_DEFAULT = [5.0, 25.0, 50.0, 100.0, 250.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda_high', type=float, default=4.0)
    p.add_argument('--t_train',     type=int,   default=8000)
    p.add_argument('--n_runs',      type=int,   default=1)
    p.add_argument('--v_grid',      type=str,   default=None,
                   help='Comma-separated V values, default {5, 25, 50, 100, 250}')
    return p.parse_args()


def make_config(lh: float, v: float, t_train: int, n_runs: int) -> Config:
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA      = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    cfg = _C()
    cfg.T_TRAIN = t_train
    cfg.N_EVAL_RUNS = n_runs
    # Paper-final knobs
    cfg.BETA      = 0.02
    cfg.W_DONE    = 10.0
    cfg.W_HL      = 2.0
    cfg.W_TIMEOUT = 5.0
    cfg.W_REJECT  = 5.0
    cfg.W_QUEUE   = 0.05
    # The V being swept
    cfg.V         = v
    return cfg


def run_one_v(v: float, args, parent_dir: str):
    cfg = make_config(args.lambda_high, v, args.t_train, args.n_runs)
    tag = f'V{v:g}'
    runner = ExperimentRunner(cfg, f'SensV_{tag}')
    logger = runner.logger
    env    = SatelliteMECEnv(cfg)
    logger.info(f"=== V = {v} | λ_high = {cfg.LAMBDA_HIGH} | T_TRAIN = {cfg.T_TRAIN} ===")
    logger.info(f"BETA={cfg.BETA}, W_DONE={cfg.W_DONE}, W_HL={cfg.W_HL}, "
                f"W_TIMEOUT={cfg.W_TIMEOUT}, W_REJECT={cfg.W_REJECT}")

    runner.setup_algorithm_dir('MAPPO')

    lyapunov_default = LyapunovCalculator(cfg)
    mappo = MAPPOPolicy(cfg, lyapunov_calc=lyapunov_default, name='MAPPO')

    # Train
    logger.info("训练 MAPPO")
    env.lyapunov_calc = lyapunov_default
    runner.run_training(mappo, env)

    # Eval (only MAPPO)
    logger.info(f"评估 MAPPO：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")
    for run_idx in range(cfg.N_EVAL_RUNS):
        seeds = cfg.get_eval_seeds(run_idx)
        rec = MetricsRecorder(cfg, 'MAPPO')
        env.reset(phase='eval', seeds=seeds)
        mappo.set_eval_mode()
        iterator = (tqdm(range(cfg.T_EVAL),
                         desc=f'[V={v:g}|MAPPO] r{run_idx}', ncols=100, unit='slot')
                    if HAS_TQDM else range(cfg.T_EVAL))
        for _ in iterator:
            _, _, _, info = env.step(policy=mappo)
            rec.record_slot(info, phase='eval')
        rec.record_eval_run(run_idx)
        cr = env.get_eval_completion_rate()
        logger.info(f"  [MAPPO] r{run_idx}: CR={cr:.4f}")
        runner.recorders['MAPPO']._slot_records.extend(rec._slot_records)
        runner.recorders['MAPPO']._eval_run_records.extend(rec._eval_run_records)

    runner.save_all_results(generate_plots=False)

    s = runner.recorders['MAPPO'].get_summary()
    return {
        'V':            v,
        'lambda_high':  cfg.LAMBDA_HIGH,
        'T_TRAIN':      cfg.T_TRAIN,
        'CR':           s.get('completion_rate', {}).get('mean', 0),
        'Sat':          s.get('avg_satisfaction_rate', {}).get('mean', 0),
        'delay':        s.get('avg_e2e_delay', {}).get('mean', 0),
        'HL':           s.get('avg_health_loss', {}).get('mean', 0),
        'DoD':          s.get('avg_dod', {}).get('mean', 0),
        'queue_mb':     s.get('avg_queue_mb', {}).get('mean', 0),
        'Z':            s.get('avg_z', {}).get('mean', 0),
        'sub_dir':      runner.base_dir,
    }


def load_baselines_reference(full_dir: str):
    """Load fixed-V baseline values from the canonical final_b run."""
    out = {}
    for pol in ['LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']:
        p = os.path.join(full_dir, pol, 'eval_runs.json')
        if not os.path.exists(p): continue
        r = json.load(open(p))[0]
        out[pol] = {
            'CR':       r['completion_rate'],
            'Sat':      r.get('avg_satisfaction_rate', 0),
            'delay':    r.get('avg_e2e_delay', 0),
            'HL':       r['avg_health_loss'],
            'DoD':      r['avg_dod'],
            'queue_mb': r.get('avg_queue_mb', 0),
        }
    return out


def main():
    args = parse_args()
    v_grid = ([float(v) for v in args.v_grid.split(',')]
              if args.v_grid else V_GRID_DEFAULT)

    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'results', time.strftime('%Y%m%d_%H%M%S') + '_SensV_paper')
    os.makedirs(parent_dir, exist_ok=True)
    print(f"敏感性分析父目录：{parent_dir}")

    # Load reference baseline values (from canonical paper final run)
    BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'results', '20260607_135453_MAPPO_lh4.0_final_b')
    baselines_ref = load_baselines_reference(BASELINE_DIR) if os.path.exists(BASELINE_DIR) else {}

    all_results = []
    for v in v_grid:
        print(f"\n{'='*70}\n  V = {v}\n{'='*70}\n")
        t0 = time.time()
        try:
            res = run_one_v(v, args, parent_dir)
            all_results.append(res)
        except Exception as e:
            print(f"[V={v}] FAILED: {e}")
            continue
        print(f"\n[V={v}] 完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
        out = {'mappo_sweep': all_results, 'baselines_reference': baselines_ref}
        with open(os.path.join(parent_dir, 'sensitivity_v_results.json'), 'w') as f:
            json.dump(out, f, indent=2)

    # Summary print
    print("\n" + "="*100)
    print("V Sensitivity Summary (LyaMAPPO at λ_high=4.0, 8K training)")
    print("="*100)
    print(f"{'V':>8} {'CR':>9} {'Sat':>9} {'delay':>9} {'HL':>13} {'DoD':>9} {'Q(MB)':>9} {'Z':>11}")
    print('-'*100)
    for r in all_results:
        print(f"{r['V']:>8.1f} {r['CR']:>9.4f} {r['Sat']:>9.4f} "
              f"{r['delay']:>9.3f} {r['HL']:>13.3e} {r['DoD']:>9.4f} "
              f"{r['queue_mb']:>9.2f} {r['Z']:>11.4e}")

    print(f"\n全部完成！结果目录：{parent_dir}")
    print(f"接下来：python plot_sensitivity_v.py --dir {parent_dir}")


if __name__ == '__main__':
    main()
