"""TD3-Sched 训练脚本（Huang TMC2024 小尺度调度器复现，无电池奖励口径）。"""
import argparse, time, numpy as np
from core import Config, SatelliteMECEnv
from baselines.td3_sched import TD3SchedPolicy

def make_cfg(lh):
    class _C(Config):
        LAMBDA_HIGH = lh
        LAMBDA = lh * Config.LAMBDA_HIGH_RATIO + Config.LAMBDA_LOW * (1 - Config.LAMBDA_HIGH_RATIO)
    c = _C(); c.N_EVAL_RUNS = 1; return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t_train', type=int, default=16000)
    ap.add_argument('--eval_every', type=int, default=4000)
    ap.add_argument('--tag', type=str, default='16K')
    args = ap.parse_args()
    cfg = make_cfg(4.0)
    cfg.W_HL = 0.0                              # outcome 也去电池：Huang 成本无电池健康项
    env = SatelliteMECEnv(cfg)
    td3 = TD3SchedPolicy(cfg, env)
    env.lyapunov_calc = td3.cost_calc           # 无电池奖励口径（守 HL 护城河）
    print(f'[TD3] 训练 {args.t_train} 槽, λ=4, 无电池奖励(含outcome), '
          f'eval_every={args.eval_every}', flush=True)
    env.reset(phase='train'); td3.set_train_mode()
    t0 = time.time()
    for t in range(args.t_train):
        td3.run_step(env)
        if (t+1) % args.eval_every == 0:
            cr, dod, hl = td3.quick_eval(env, n_slots=800)
            el = time.time()-t0
            print(f'[TD3] slot {t+1}/{args.t_train} | CR={cr:.3f} DoD={dod:.3f} HL={hl:.3e} '
                  f'| {el:.0f}s ({(t+1)/max(el,1):.0f} slot/s)', flush=True)
    path = f'checkpoints/TD3Sched_lh4_{args.tag}'
    td3.save(path)
    print(f'[TD3] DONE 总耗时 {time.time()-t0:.0f}s, 模型存 {path}', flush=True)

if __name__ == '__main__':
    main()
