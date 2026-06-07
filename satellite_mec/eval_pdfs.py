"""eval_pdfs.py
================
Load a saved MAPPO checkpoint, run all 5 policies for T_EVAL slots at
λ_high=4, and render 8 PDF/CDF subplots covering:

  Delay PDFs:
    A1  PDF of completed-task end-to-end delays
    A2  PDF including timed-out tasks (timeout virtual delay = deadline)
    A3  CDF of completed-task delays
    A4  CDF of all admitted tasks

  Satisfaction PDFs:
    B1  PDF of per-slot satisfaction rate distribution
    B2  PDF of slack ratio (deadline − delay) / deadline for completed tasks
    B3  CDF of per-slot satisfaction rate
    B4  CDF of slack ratio

Each subplot overlays the 5 policies as KDE curves.

Usage
-----
    python eval_pdfs.py --ckpt results/<DIR>/MAPPO/model
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from core import Config, SatelliteMECEnv, LyapunovCalculator
from baselines import (LocalOnlyPolicy, GreedyDelayPolicy,
                       LyapunovGreedyPolicy, MHSPOPolicy)
from training import MAPPOPolicy


POLICIES = ['MAPPO', 'LocalOnly', 'GreedyDelay', 'LyapunovGreedy', 'MHSPO']
COLORS   = {
    'MAPPO':          'C3',
    'MHSPO':          'C0',
    'LyapunovGreedy': 'C2',
    'GreedyDelay':    'C1',
    'LocalOnly':      'C7',
}
LINESTYLES = {
    'MAPPO': '-',
    'MHSPO': '--',
    'LyapunovGreedy': ':',
    'GreedyDelay': '-.',
    'LocalOnly': '-',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda_high', type=float, default=4.0)
    p.add_argument('--ckpt',        type=str, required=True)
    p.add_argument('--out_dir',     type=str, default=None)
    return p.parse_args()


def make_config(lh):
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
        N_EVAL_RUNS = 1
    return _C()


def run_policy(env, policy, t_eval, label):
    """Return dict of accumulated per-task / per-slot stats."""
    delays = []          # completed e2e delays
    done_ddls = []       # deadlines of completed tasks
    timeout_ddls = []    # deadlines of timed-out tasks (compute_timeout only)
    sat_rates = []       # slot satisfaction rate
    iterator = (tqdm(range(t_eval), desc=label, ncols=100, unit='slot')
                if HAS_TQDM else range(t_eval))
    for _ in iterator:
        _, _, _, info = env.step(policy=policy)
        delays.extend(info.get('slot_e2e_delays', []))
        done_ddls.extend(info.get('slot_done_deadlines', []))
        timeout_ddls.extend(info.get('slot_timeout_deadlines', []))
        sat_rates.append(info.get('slot_satisfaction_rate', 0.0))
    return {
        'delays':       np.array(delays),
        'done_ddls':    np.array(done_ddls),
        'timeout_ddls': np.array(timeout_ddls),
        'sat_rates':    np.array(sat_rates),
        'cr':           float(env.get_eval_completion_rate()),
    }


def kde_curve(samples, x_range, bw=None):
    """Return (x, y) for KDE smooth plot."""
    samples = np.asarray(samples)
    if len(samples) < 2 or samples.std() < 1e-9:
        # not enough variance: return spike at mean
        x = np.linspace(*x_range, 200)
        y = np.zeros_like(x)
        return x, y
    kde = gaussian_kde(samples, bw_method=bw)
    x = np.linspace(*x_range, 400)
    y = kde(x)
    return x, y


def cdf_curve(samples, x_range):
    samples = np.sort(np.asarray(samples))
    if len(samples) == 0:
        return np.linspace(*x_range, 2), np.zeros(2)
    # extend to plot range
    xs = np.concatenate([[x_range[0]], samples, [x_range[1]]])
    cum = np.concatenate([[0.0], np.linspace(1.0/len(samples), 1.0, len(samples)), [1.0]])
    return xs, cum


def main():
    args = parse_args()
    cfg  = make_config(args.lambda_high)
    env  = SatelliteMECEnv(cfg)

    print(f"\n=== Evaluating 5 policies for {cfg.T_EVAL} slots @ λ_high={args.lambda_high} ===")

    mappo = MAPPOPolicy(cfg, name='MAPPO')
    mappo.load(args.ckpt)
    local_only   = LocalOnlyPolicy(cfg, env)
    greedy_delay = GreedyDelayPolicy(cfg, env)
    lya_greedy   = LyapunovGreedyPolicy(cfg, env)
    mhspo        = MHSPOPolicy(cfg, env, rho_d=1.0, rho_e=1.0, V_lyapunov=10.0)

    # Warm up MHSPO DOGD
    print('Warming up MHSPO DOGD...')
    env.reset(phase='train')
    iter_ = (tqdm(range(cfg.T_WARMUP), desc='warmup MHSPO', ncols=100, unit='slot')
             if HAS_TQDM else range(cfg.T_WARMUP))
    for _ in iter_:
        env.step(policy=mhspo)

    pol_objs = {'MAPPO': mappo, 'LocalOnly': local_only,
                'GreedyDelay': greedy_delay, 'LyapunovGreedy': lya_greedy,
                'MHSPO': mhspo}

    seeds = cfg.get_eval_seeds(0)
    results = {}
    for name in POLICIES:
        pol = pol_objs[name]
        env.reset(phase='eval', seeds=seeds)
        pol.set_eval_mode()
        results[name] = run_policy(env, pol, cfg.T_EVAL, name)
        print(f"  {name:<14} CR={results[name]['cr']:.4f}  "
              f"completed={len(results[name]['delays'])} "
              f"timeouts={len(results[name]['timeout_ddls'])} "
              f"sat_rate_mean={results[name]['sat_rates'].mean():.4f}")

    # Build 8 subplots in 2 main figures
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), '..', '..', 'figures_pdf')
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure A: Delay PDFs / CDFs ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x_delay = (0, 12)
    for ax, mode in zip(axes.flat, ['pdf_completed', 'pdf_all', 'cdf_completed', 'cdf_all']):
        for name in POLICIES:
            r = results[name]
            if mode == 'pdf_completed' or mode == 'cdf_completed':
                samples = r['delays']
            else:
                # all admitted: completed delays + timeout deadlines (worst-case delay)
                samples = np.concatenate([r['delays'], r['timeout_ddls']])
            if len(samples) == 0: continue
            if mode.startswith('pdf'):
                x, y = kde_curve(samples, x_delay)
            else:
                x, y = cdf_curve(samples, x_delay)
            ax.plot(x, y, label=name, color=COLORS[name],
                    linestyle=LINESTYLES[name], linewidth=1.8, alpha=0.85)
        if mode == 'pdf_completed':
            ax.set_title('Delay PDF (completed tasks only)')
            ax.set_ylabel('Density')
        elif mode == 'pdf_all':
            ax.set_title('Delay PDF (all admitted: timeouts at deadline)')
            ax.set_ylabel('Density')
        elif mode == 'cdf_completed':
            ax.set_title('Delay CDF (completed tasks only)')
            ax.set_ylabel('P(delay <= x)')
        else:
            ax.set_title('Delay CDF (all admitted)')
            ax.set_ylabel('P(delay <= x)')
        ax.set_xlabel('End-to-end delay (s)')
        ax.set_xlim(*x_delay)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    plt.suptitle('End-to-End Delay Distributions @ λ_high=4.0',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '09_delay_pdf_cdf.png'), dpi=150)
    plt.close()
    print('saved 09_delay_pdf_cdf.png')

    # ── Figure B: Satisfaction PDFs / CDFs ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x_rate  = (0, 1.05)
    x_slack = (-0.5, 1.05)
    for ax, mode in zip(axes.flat, ['pdf_rate', 'pdf_slack', 'cdf_rate', 'cdf_slack']):
        for name in POLICIES:
            r = results[name]
            if mode == 'pdf_rate' or mode == 'cdf_rate':
                samples = r['sat_rates']
                x_range = x_rate
            else:
                # slack ratio: (deadline - delay) / deadline
                if len(r['delays']) == 0: continue
                samples = (r['done_ddls'] - r['delays']) / np.maximum(r['done_ddls'], 1e-6)
                x_range = x_slack
            if len(samples) == 0: continue
            if mode.startswith('pdf'):
                x, y = kde_curve(samples, x_range)
            else:
                x, y = cdf_curve(samples, x_range)
            ax.plot(x, y, label=name, color=COLORS[name],
                    linestyle=LINESTYLES[name], linewidth=1.8, alpha=0.85)
        if mode == 'pdf_rate':
            ax.set_title('Per-slot Satisfaction Rate PDF')
            ax.set_xlabel('Slot satisfaction rate'); ax.set_ylabel('Density')
        elif mode == 'pdf_slack':
            ax.set_title('Slack Ratio PDF\n(deadline - delay) / deadline,  completed tasks')
            ax.set_xlabel('Slack ratio (1.0 = whole budget left)')
            ax.set_ylabel('Density')
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        elif mode == 'cdf_rate':
            ax.set_title('Per-slot Satisfaction Rate CDF')
            ax.set_xlabel('Slot satisfaction rate')
            ax.set_ylabel('P(rate <= x)')
        else:
            ax.set_title('Slack Ratio CDF')
            ax.set_xlabel('Slack ratio')
            ax.set_ylabel('P(slack <= x)')
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    plt.suptitle('User Satisfaction Distributions @ λ_high=4.0',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '10_satisfaction_pdf_cdf.png'), dpi=150)
    plt.close()
    print('saved 10_satisfaction_pdf_cdf.png')

    # Also dump raw samples for reproducibility
    dump = {name: {
        'cr':            r['cr'],
        'n_completed':   len(r['delays']),
        'n_timeout':     len(r['timeout_ddls']),
        'delay_mean':    float(r['delays'].mean()) if len(r['delays']) else 0.0,
        'delay_median':  float(np.median(r['delays'])) if len(r['delays']) else 0.0,
        'delay_p95':     float(np.percentile(r['delays'], 95)) if len(r['delays']) else 0.0,
        'sat_rate_mean': float(r['sat_rates'].mean()),
    } for name, r in results.items()}
    with open(os.path.join(out_dir, 'pdf_summary.json'), 'w') as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)
    print(f'\nAll figures + summary saved to: {out_dir}')


if __name__ == '__main__':
    main()
