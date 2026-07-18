"""MADDPG-DoD 训练脚本（Zhong et al. IoT-J 2026 忠实复现，作为 LyaMAPPO 对比 baseline）。

复现口径 V2：论文原始 4 维 obs + DoD drift-plus-penalty 奖励 + concat 中心化 critic +
Gumbel-Softmax，共享本仓库 Lyapunov-DVFS 物理基底保证对比公平。奖励由策略自行计算
（_paper_team_reward），不使用 env 的 outcome-aware 注入。

用法
----
    python train_maddpg_dod.py --t_train 32000 --tag 32K --upsilon 1000
"""
import argparse
import time

import numpy as np

from core import Config, SatelliteMECEnv
from baselines.maddpg_dod import MADDPGDoDPolicy


def make_cfg(lh: float) -> Config:
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    c = _C()
    c.N_EVAL_RUNS = 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t_train', type=int, default=32000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--tag', type=str, default='32K')
    ap.add_argument('--cr_floor', type=float, default=0.32,
                    help='best-checkpoint 的 CR 下限（保证不选到吞吐塌缩点）')
    # 默认值 = 冒烟校准后的定稿（DoD-primary，队列/时延次级，λ=4）
    ap.add_argument('--upsilon', type=float, default=50.0, help='论文 Lyapunov 控制参数 υ（DoD 权重）')
    ap.add_argument('--w_queue', type=float, default=15.0)
    ap.add_argument('--w_delay', type=float, default=0.02)
    ap.add_argument('--t_max_delay', type=float, default=3.0, help='论文式固定时延约束 T_max(s)')
    ap.add_argument('--y_max', type=float, default=10.0, help='时延虚拟队列上限')
    ap.add_argument('--expl_noise', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cfg = make_cfg(4.0)
    env = SatelliteMECEnv(cfg)
    maddpg = MADDPGDoDPolicy(cfg, env, upsilon=args.upsilon,
                             w_queue=args.w_queue, w_delay=args.w_delay,
                             t_max_delay=args.t_max_delay, y_max=args.y_max,
                             expl_noise=args.expl_noise, seed=args.seed)
    print(f'[MADDPG-DoD] 训练 {args.t_train} 槽, λ=4, υ={args.upsilon}, '
          f'concat critic dim={maddpg.joint_obs_dim + maddpg.joint_act_dim}, '
          f'eval_every={args.eval_every}', flush=True)

    # best-checkpoint 选择：论文目标 = 最小化 DoD s.t. 队列稳定；
    # 取满足 CR≥cr_floor 的最低-DoD 快评点为定稿（避免 MADDPG 训练不稳落在平庸/塌缩点）
    path = f'checkpoints/MADDPG_DoD_lh4_{args.tag}'
    best_dod = float('inf'); best_step = -1; saved_best = False

    env.reset(phase='train'); maddpg.set_train_mode()
    t0 = time.time()
    for t in range(args.t_train):
        maddpg.run_step(env)
        if (t + 1) % args.eval_every == 0:
            cr, dod, hl = maddpg.quick_eval(env, n_slots=800)
            el = time.time() - t0
            tag_best = ''
            if cr >= args.cr_floor and dod < best_dod:
                best_dod, best_step = dod, t + 1; saved_best = True
                maddpg.save(path, extra_info={'t_train': args.t_train, 'lambda_high': 4.0,
                            'best_step': best_step, 'best_dod': dod, 'best_cr': cr,
                            'select': 'min-DoD s.t. CR>=%.2f' % args.cr_floor})
                tag_best = f'  ** NEW BEST (DoD={dod:.4f}, CR={cr:.3f}) saved'
            print(f'[MADDPG-DoD] slot {t+1}/{args.t_train} | CR={cr:.3f} DoD={dod:.3f} '
                  f'HL={hl:.3e} | {el:.0f}s ({(t+1)/max(el,1):.0f} slot/s){tag_best}', flush=True)

    if not saved_best:                       # 兜底：无点达 CR 下限则存 final
        maddpg.save(path, extra_info={'t_train': args.t_train, 'lambda_high': 4.0,
                    'note': 'fallback-final (no eval met cr_floor)'})
        print('[MADDPG-DoD] 警告：无快评点达 CR 下限，已存 final 兜底', flush=True)
    print(f'[MADDPG-DoD] DONE 总耗时 {time.time()-t0:.0f}s | best DoD={best_dod:.4f} '
          f'@ step {best_step} | 模型存 {path}', flush=True)


if __name__ == '__main__':
    main()
