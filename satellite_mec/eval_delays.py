"""eval_delays.py
==================
Evaluate all 5 policies and extract real end-to-end delay statistics
(mean / median / p95 / p99) — these are NOT in eval_runs.json by default.

Loads MAPPO checkpoint from --ckpt path, runs T_EVAL slots at λ_high=4
for each policy, accumulates per-task delays from slot_e2e_delays.
"""
import argparse
import json
import os
import sys
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv, LyapunovCalculator
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)
from training import MAPPOPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda_high', type=float, default=4.0)
    p.add_argument('--ckpt',        type=str, required=True,
                   help='MAPPO model dir containing actor.pth')
    p.add_argument('--n_runs',      type=int, default=1)
    return p.parse_args()


def make_config(lh: float) -> Config:
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
        N_EVAL_RUNS = 1
    return _C()


def eval_policy(env, policy, t_eval, n_sats, label):
    """Run T_EVAL slots, accumulate e2e delays."""
    all_delays = []
    iterator = (tqdm(range(t_eval), desc=label, ncols=100, unit='slot')
                if HAS_TQDM else range(t_eval))
    for _ in iterator:
        _, _, _, info = env.step(policy=policy)
        all_delays.extend(info.get('slot_e2e_delays', []))
    return np.array(all_delays)


def warmup_mhspo(env, mhspo, t_warmup=5400):
    """Warm up MHSPO DOGD predictor in train phase."""
    env.reset(phase='train')
    mhspo.set_train_mode() if hasattr(mhspo, 'set_train_mode') else None
    iter_ = (tqdm(range(t_warmup), desc='预热MHSPO', ncols=100, unit='slot')
             if HAS_TQDM else range(t_warmup))
    for _ in iter_:
        env.step(policy=mhspo)


def main():
    args = parse_args()
    cfg  = make_config(args.lambda_high)
    env  = SatelliteMECEnv(cfg)

    print(f"\n=== λ_high={args.lambda_high}, T_EVAL={cfg.T_EVAL}, n_runs={args.n_runs} ===\n")

    # 创建 5 个策略
    mappo        = MAPPOPolicy(cfg, name='MAPPO')
    mappo.load(args.ckpt)
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)

    # MHSPO 需要预热
    warmup_mhspo(env, mhspo)

    policies = [mappo, local_only, greedy_delay, lya_greedy, mhspo]
    results = {}

    for run_idx in range(args.n_runs):
        seeds = cfg.get_eval_seeds(run_idx)
        for policy in policies:
            env.reset(phase='eval', seeds=seeds)
            policy.set_eval_mode()
            delays = eval_policy(env, policy, cfg.T_EVAL,
                                 cfg.N_SATS, f'[{policy.name}] r{run_idx}')
            cr  = env.get_eval_completion_rate()
            results.setdefault(policy.name, {
                'delays': [], 'cr_runs': []
            })
            results[policy.name]['delays'].extend(delays.tolist())
            results[policy.name]['cr_runs'].append(cr)

    # 报告
    print('\n' + '='*82)
    print(f"  λ_high={args.lambda_high} 端到端延迟统计 (基于完成任务的真实延迟)")
    print('='*82)
    print(f"{'策略':<16}{'CR':>8}{'#完成':>8}{'mean(s)':>10}{'median':>10}"
          f"{'p95':>10}{'p99':>10}{'max':>10}")
    print('-'*82)
    for pol in ['MAPPO','LocalOnly','GreedyDelay','LyapunovGreedy','MHSPO']:
        d = np.array(results[pol]['delays'])
        cr = np.mean(results[pol]['cr_runs'])
        if len(d) == 0:
            print(f"  {pol:<14}{cr:>8.4f}{0:>8d}  no completed tasks")
            continue
        print(f"  {pol:<14}{cr:>8.4f}{len(d):>8d}"
              f"{np.mean(d):>10.3f}{np.median(d):>10.3f}"
              f"{np.percentile(d, 95):>10.3f}{np.percentile(d, 99):>10.3f}"
              f"{np.max(d):>10.3f}")

    # 保存原始延迟样本（供画 PDF）
    out_path = os.path.join(os.path.dirname(args.ckpt), '..', '..', 'delay_stats.json')
    with open(out_path, 'w') as f:
        json.dump({pol: {
            'cr_mean': float(np.mean(results[pol]['cr_runs'])),
            'delay_mean':   float(np.mean(results[pol]['delays'])) if results[pol]['delays'] else None,
            'delay_median': float(np.median(results[pol]['delays'])) if results[pol]['delays'] else None,
            'delay_p95':    float(np.percentile(results[pol]['delays'], 95)) if results[pol]['delays'] else None,
            'delay_p99':    float(np.percentile(results[pol]['delays'], 99)) if results[pol]['delays'] else None,
            'n_samples':    len(results[pol]['delays']),
        } for pol in results}, f, indent=2, ensure_ascii=False)
    print(f"\n延迟统计已保存至: {out_path}")


if __name__ == '__main__':
    main()
