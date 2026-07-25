"""MADDPG-DoD 训练脚本（Zhong 2026 学习器,公平起跑线口径,作为 LyaMAPPO 对比 baseline）。

口径（与爸爸定稿）
------------------
- 与所有学习型基线相同的 env / DVFS 基底 / 完成激励（不塌吞吐,公平对比）;
- 忠实论文电池处理:**DoD-aware**（use_dod_penalty=True）,**无 HL 半衰期项**
  （use_battery_loss=False,且 W_HL=0）——HL 虚拟队列是 LyaMAPPO 原创;
- 学习器 = MADDPG（Gumbel-Softmax 离散 actor + 245 维中心化 critic + 目标网软更新）。

用法
----
    python train_maddpg_dod.py --t_train 32000 --tag 32K
"""
import argparse
import time

from core import Config, SatelliteMECEnv
from core.lyapunov import LyapunovCalculator
from baselines.maddpg_dod import MADDPGDoDPolicy


def make_cfg(lh: float) -> Config:
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    c = _C()
    c.N_EVAL_RUNS = 1
    c.W_HL = 0.0                    # 忠实论文:无 HL 半衰期奖励项
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t_train', type=int, default=32000)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--tag', type=str, default='32K')
    ap.add_argument('--expl_noise', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--w_done', type=float, default=None,
                    help='override completion-reward weight (default: config 5.0). '
                         'Lower (e.g. 1.0) drops into the low-throughput attractor '
                         'for the throughput-HL trade-off frontier.')
    ap.add_argument('--eta', type=float, default=None,
                    help='override DoD virtual-queue penalty weight (default: config 0.5). '
                         'Higher brings the DoD/energy penalty to parity with the '
                         'completion incentive, so the policy actually reduces HL.')
    ap.add_argument('--zhong', action='store_true',
                    help='忠实 Zhong eq41 reward: −Q·l −Y·(T−Tmax) −υ·D，队列漂移驱动吞吐、无 W_DONE。')
    ap.add_argument('--upsilon', type=float, default=None,
                    help='Zhong DoD 惩罚权重 υ（--zhong 下 DoD↔service 旋钮，default config 1.0）')
    ap.add_argument('--t_max', type=float, default=None,
                    help='Zhong 时延虚拟队列 Tmax(s)（default config 6.0）')
    args = ap.parse_args()

    cfg = make_cfg(4.0)
    if args.w_done is not None:
        cfg.W_DONE = args.w_done
    if args.eta is not None:
        cfg.ETA = args.eta
    if args.upsilon is not None:
        cfg.UPSILON = args.upsilon
    if args.t_max is not None:
        cfg.T_MAX_DELAY = args.t_max
    env = SatelliteMECEnv(cfg)
    # DoD-aware 但无 HL 的物理基底（忠实论文:优化 DoD,不优化 HL 半衰期）
    env.lyapunov_calc = LyapunovCalculator(cfg, use_dod_penalty=True, use_battery_loss=False)

    maddpg = MADDPGDoDPolicy(cfg, env, expl_noise=args.expl_noise, seed=args.seed,
                             zhong_reward=args.zhong, upsilon=cfg.UPSILON)
    if args.zhong:
        print(f'[MADDPG-DoD/Zhong] 训练 {args.t_train} 槽, λ=4, 忠实 eq41 reward '
              f'(−Q·l −Y·(T−Tmax) −υ·D, 无 W_DONE), υ={cfg.UPSILON}, Tmax={cfg.T_MAX_DELAY}, '
              f'critic={maddpg.c_dim}维, eval_every={args.eval_every}', flush=True)
    else:
        print(f'[MADDPG-DoD] 训练 {args.t_train} 槽, λ=4, 公平起跑线(env奖励+完成激励, '
              f'DoD-aware 无HL), W_DONE={cfg.W_DONE}, ETA={cfg.ETA}, critic={maddpg.c_dim}维, '
              f'eval_every={args.eval_every}', flush=True)

    path = f'checkpoints/MADDPG_DoD_lh4_{args.tag}'
    best_cr = -1.0; best_step = -1; saved = False

    env.reset(phase='train'); maddpg.set_train_mode()
    t0 = time.time()
    for t in range(args.t_train):
        maddpg.run_step(env)
        if (t + 1) % args.eval_every == 0:
            cr, dod, hl = maddpg.quick_eval(env, n_slots=800)
            el = time.time() - t0
            tag_best = ''
            if cr > best_cr:                         # 竞争型基线:取最高满意度(CR)代表
                best_cr, best_step = cr, t + 1; saved = True
                maddpg.save(path, extra_info={'t_train': args.t_train, 'lambda_high': 4.0,
                            'best_step': best_step, 'best_cr': cr, 'best_dod': dod,
                            'select': 'max-CR (fair-footing competitive baseline)'})
                tag_best = f'  ** NEW BEST (CR={cr:.3f}) saved'
            print(f'[MADDPG-DoD] slot {t+1}/{args.t_train} | CR={cr:.3f} DoD={dod:.3f} '
                  f'HL={hl:.3e} | {el:.0f}s ({(t+1)/max(el,1):.0f} slot/s){tag_best}', flush=True)

    if not saved:
        maddpg.save(path, extra_info={'t_train': args.t_train, 'note': 'fallback-final'})
    print(f'[MADDPG-DoD] DONE 总耗时 {time.time()-t0:.0f}s | best CR={best_cr:.3f} '
          f'@ step {best_step} | 模型存 {path}', flush=True)


if __name__ == '__main__':
    main()
