"""eval_multi_runs.py
==========================
Load LyaMAPPO checkpoint and evaluate with multiple eval-seed runs to
compute mean ± CI95 for each metric. Compare against MHSPO (and
optionally all baselines) for fair statistical comparison.

Usage:
    python eval_multi_runs.py --ckpt checkpoints/LyaMAPPO_lh4_32K --n_runs 3
    python eval_multi_runs.py --ckpt ... --n_runs 5 --include_baselines
"""
import argparse
import json
import os
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)
from training import MAPPOPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str, required=True)
    p.add_argument('--lambda_high', type=float, default=4.0)
    p.add_argument('--n_runs',      type=int,   default=3)
    p.add_argument('--include_baselines', action='store_true')
    return p.parse_args()


def make_config(lh, n_runs):
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    cfg = _C()
    cfg.N_EVAL_RUNS = n_runs
    cfg.BETA      = 0.02
    cfg.W_DONE    = 10.0
    cfg.W_HL      = 2.0
    cfg.W_TIMEOUT = 5.0
    cfg.W_REJECT  = 5.0
    cfg.W_QUEUE   = 0.05
    return cfg


def collect_metrics(env, policy, label, n_slots):
    delays, sat_rates, slot_hl, slot_dod, slot_q = [], [], [], [], []
    iterator = (tqdm(range(n_slots), desc=label, ncols=100, unit='slot')
                if HAS_TQDM else range(n_slots))
    for _ in iterator:
        _, _, _, info = env.step(policy=policy)
        delays.extend(info.get('slot_e2e_delays', []))
        sat_rates.append(info.get('slot_satisfaction_rate', 0.0))
        slot_hl.append(info.get('avg_health_loss', 0.0))
        slot_dod.append(info.get('avg_dod', 0.0))
        slot_q.append(info.get('total_queue_size', 0.0))
    return {
        'cr':          float(env.get_eval_completion_rate()),
        'sat_mean':    float(np.mean(sat_rates)),
        'delay_mean':  float(np.mean(delays)) if delays else 0.0,
        'hl_per_slot': float(np.mean(slot_hl)),
        'dod':         float(np.mean(slot_dod)),
        'queue_mb':    float(np.mean(slot_q)) / 1e6,
    }


def run_n_evals(env, policy, cfg, name):
    print(f'\n--- evaluating {name} with n_runs={cfg.N_EVAL_RUNS} ---')
    per_run = []
    for run_idx in range(cfg.N_EVAL_RUNS):
        seeds = cfg.get_eval_seeds(run_idx)
        env.reset(phase='eval', seeds=seeds)
        policy.set_eval_mode()
        m = collect_metrics(env, policy, f'[{name}] r{run_idx}', cfg.T_EVAL)
        per_run.append(m)
        print(f"  r{run_idx}: CR={m['cr']:.4f}  HL={m['hl_per_slot']:.3e}  "
              f"delay={m['delay_mean']:.3f}s  DoD={m['dod']:.4f}")
    agg = {}
    for k in per_run[0]:
        v = [r[k] for r in per_run]
        m = float(np.mean(v)); s = float(np.std(v))
        ci = 1.96 * s / np.sqrt(cfg.N_EVAL_RUNS) if cfg.N_EVAL_RUNS > 1 else 0.0
        agg[k] = {'mean': m, 'std': s, 'ci95': ci, 'values': v}
    return agg


def warmup_mhspo(env, mhspo, t_warmup):
    print(f'Warming up MHSPO for {t_warmup} slots...')
    env.reset(phase='train')
    iter_ = (tqdm(range(t_warmup), desc='warmup', ncols=100, unit='slot')
             if HAS_TQDM else range(t_warmup))
    for _ in iter_:
        env.step(policy=mhspo)


def main():
    args = parse_args()
    cfg = make_config(args.lambda_high, args.n_runs)
    env = SatelliteMECEnv(cfg)

    mappo = MAPPOPolicy(cfg, name='LyaMAPPO')
    mappo.load(args.ckpt)

    if args.include_baselines:
        local_only   = LocalOnlyPolicy(cfg, env)
        greedy_delay = GreedyDelayPolicy(cfg, env)
        lya_greedy   = LyapunovGreedyPolicy(cfg, env)
        mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)
        warmup_mhspo(env, mhspo, cfg.T_WARMUP)
        policies = [('LyaMAPPO', mappo), ('LocalOnly', local_only),
                    ('GreedyDelay', greedy_delay),
                    ('LyapunovGreedy', lya_greedy), ('MHSPO', mhspo)]
    else:
        mhspo = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)
        warmup_mhspo(env, mhspo, cfg.T_WARMUP)
        policies = [('LyaMAPPO', mappo), ('MHSPO', mhspo)]

    results = {}
    for name, pol in policies:
        results[name] = run_n_evals(env, pol, cfg, name)

    # Summary
    print('\n' + '=' * 102)
    print(f"n_runs = {cfg.N_EVAL_RUNS}  (mean ± CI95)")
    print('=' * 102)
    keys   = ['cr', 'sat_mean', 'delay_mean', 'hl_per_slot', 'dod', 'queue_mb']
    labels = ['CR', 'Sat', 'Delay(s)', 'HL/slot', 'DoD', 'Q(MB)']
    print(f"{'policy':<14}", end='')
    for L in labels: print(f"{L:>17}", end='')
    print()
    print('-' * 102)
    for name, _ in policies:
        print(f"{name:<14}", end='')
        for k in keys:
            m  = results[name][k]['mean']
            ci = results[name][k]['ci95']
            if k == 'hl_per_slot':
                print(f"  {m:.3e}±{ci:.0e}", end='')
            elif k in ('cr', 'sat_mean', 'dod'):
                print(f"   {m:.4f}±{ci:.4f}", end='')
            elif k == 'queue_mb':
                print(f"  {m:8.2f}±{ci:5.2f}", end='')
            else:
                print(f"   {m:7.3f}±{ci:5.3f}", end='')
        print()

    out_path = os.path.join(os.path.dirname(args.ckpt), '..',
                            f'multi_run_eval_n{cfg.N_EVAL_RUNS}.json')
    with open(out_path, 'w') as f:
        json.dump({'n_runs': cfg.N_EVAL_RUNS, 'results': results}, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
