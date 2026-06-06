"""
evaluation/runner.py
====================
实验流程编排器（预热 / 训练 / 评估 / 保存）。

使用示例
--------
    runner = ExperimentRunner(cfg, 'BaselinesOnly', debug=False)
    runner.setup_algorithm_dir('LocalOnly')
    runner.run_warmup(env, policy=greedy_policy)
    runner.run_evaluation([local_only, greedy], env)
    runner.save_all_results()
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from evaluation.recorder import MetricsRecorder

if TYPE_CHECKING:
    from core.config import Config
    from core.env import SatelliteMECEnv
    from interfaces import PolicyInterface


def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class ExperimentRunner:
    """
    实验流程编排器。

    职责
    ----
    - 管理实验目录结构
    - 提供 run_warmup / run_training / run_evaluation 标准流程
    - 聚合多个策略的结果并保存/绘图
    """

    def __init__(self, config: "Config", experiment_name: str, debug: bool = False):
        self.cfg             = config
        self.experiment_name = experiment_name
        self.debug           = debug

        if debug:
            self.cfg.T_TRAIN     = self.cfg.DEBUG_T_TRAIN
            self.cfg.T_WARMUP    = self.cfg.DEBUG_T_WARMUP
            self.cfg.T_EVAL      = self.cfg.DEBUG_T_EVAL
            self.cfg.T_CALIBRATE = self.cfg.DEBUG_T_CALIBRATE
            self.cfg.N_EVAL_RUNS = self.cfg.DEBUG_N_EVAL_RUNS
            self.cfg.EVAL_INTERVAL = 5
            print(f"[Runner] DEBUG 模式：T_TRAIN={self.cfg.T_TRAIN}, "
                  f"T_EVAL={self.cfg.T_EVAL}, N_RUNS={self.cfg.N_EVAL_RUNS}")

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results', f'{ts}_{experiment_name}')
        os.makedirs(self.base_dir, exist_ok=True)
        self.logger     = setup_logger('runner', self.base_dir)
        self.recorders: Dict[str, MetricsRecorder] = {}
        self.result_dirs: Dict[str, str] = {}
        self.logger.info(f"实验目录：{self.base_dir}")

    def setup_algorithm_dir(self, algorithm_name: str) -> str:
        alg_dir = os.path.join(self.base_dir, algorithm_name)
        os.makedirs(os.path.join(alg_dir, 'figures'), exist_ok=True)
        self.result_dirs[algorithm_name] = alg_dir
        self.recorders[algorithm_name]   = MetricsRecorder(self.cfg, algorithm_name)
        return alg_dir

    def calibrate_l_max(self, env: "SatelliteMECEnv") -> None:
        cfg  = self.cfg
        t_cal = cfg.DEBUG_T_CALIBRATE if self.debug else cfg.T_CALIBRATE
        self.logger.info(f"校准 L_MAX：{t_cal} 时隙…")
        env.reset(phase='warmup', seeds={'task': cfg.SEED+500,
                                         'task_param': cfg.SEED+501,
                                         'dod_init':   cfg.SEED+502})
        all_hl = []
        iterator = (tqdm(range(t_cal), desc='校准L_MAX', ncols=80, unit='slot')
                    if HAS_TQDM else range(t_cal))
        for _ in iterator:
            _, _, _, info = env.step({})
            hl = info.get('avg_health_loss', 0.0)
            if hl > 0:
                all_hl.append(hl)
        if not all_hl:
            self.logger.warning("校准失败，保持默认 L_MAX"); return
        l_max_new = float(np.percentile(all_hl, 95))
        self.logger.info(f"L_MAX：{cfg.L_MAX:.4e} → {l_max_new:.4e}")
        cfg.L_MAX = l_max_new; cfg.L_MAX_CALIBRATED = True

    def run_training(self, policy, env: "SatelliteMECEnv") -> None:
        """训练 MAPPOPolicy（需要 training 模块）。"""
        cfg = self.cfg
        recorder = self.recorders[policy.name]
        alg_dir  = self.result_dirs[policy.name]
        self.logger.info(f"[{policy.name}] 训练：{cfg.T_TRAIN} 时隙")
        env.reset(phase='train'); policy.set_train_mode()
        episode_idx = 0; start = time.time()
        checkpt_interval = max(cfg.K_ROLLOUT * 100, 1000)
        iterator = (tqdm(range(cfg.T_TRAIN), desc=f'训练[{policy.name}]',
                         ncols=100, unit='slot')
                    if HAS_TQDM else range(cfg.T_TRAIN))
        for t in iterator:
            _, done, info = policy.run_step(env)
            recorder.record_slot(info, phase='train')
            if done:
                recorder.record_episode(episode_idx); episode_idx += 1
                if (policy.trainer.update_count > 0
                        and policy.trainer.update_count % cfg.EVAL_INTERVAL == 0):
                    cr, avg_dod, _ = policy.quick_eval(env, n_slots=min(cfg.T_EVAL, 1000))
                    if HAS_TQDM:
                        iterator.set_postfix({'CR': f'{cr:.3f}', 'DoD': f'{avg_dod:.3f}'})
            if (t + 1) % checkpt_interval == 0:
                policy.save(os.path.join(alg_dir, 'checkpoints', f'step_{t+1}'))
        self.logger.info(f"[{policy.name}] 训练完成 {time.time()-start:.1f}s")
        policy.save(os.path.join(alg_dir, 'model'))

    def run_warmup(self, env: "SatelliteMECEnv",
                   policy: Optional["PolicyInterface"] = None) -> None:
        cfg = self.cfg
        name = policy.name if policy else '空动作'
        self.logger.info(f"预热：{cfg.T_WARMUP} 时隙，策略={name}")
        env.reset(phase='warmup')
        if policy is not None:
            policy.set_eval_mode()
        iterator = (tqdm(range(cfg.T_WARMUP), desc=f'预热[{name}]',
                         ncols=80, unit='slot')
                    if HAS_TQDM else range(cfg.T_WARMUP))
        for _ in iterator:
            if policy is not None:
                env.step(policy=policy)
            else:
                env.step({})

    def run_evaluation(self, policies: List["PolicyInterface"],
                       env: "SatelliteMECEnv") -> None:
        cfg = self.cfg
        self.logger.info(f"评估：{cfg.N_EVAL_RUNS} run × {cfg.T_EVAL} 时隙")
        for run_idx in range(cfg.N_EVAL_RUNS):
            seeds = cfg.get_eval_seeds(run_idx)
            for policy in policies:
                env.reset(phase='eval', seeds=seeds)
                policy.set_eval_mode()
                recorder = self.recorders[policy.name]
                iterator = (tqdm(range(cfg.T_EVAL),
                                 desc=f'评估[{policy.name}]run{run_idx}',
                                 ncols=100, unit='slot')
                            if HAS_TQDM else range(cfg.T_EVAL))
                for _ in iterator:
                    _, _, _, info = env.step(policy=policy)
                    recorder.record_slot(info, phase='eval')
                recorder.record_eval_run(run_idx)
                policy.set_train_mode()
                cr = env.get_eval_completion_rate()
                self.logger.info(f"  [{policy.name}] run{run_idx}: CR={cr:.4f}")

    def save_all_results(self, generate_plots: bool = True) -> None:
        self.logger.info("保存实验结果…")
        for alg_name, recorder in self.recorders.items():
            recorder.save(self.result_dirs[alg_name])
        if generate_plots:
            self._generate_comparison_plots()
        self.logger.info(f"完成！目录：{self.base_dir}")

    def _generate_comparison_plots(self) -> None:
        from evaluation.plotting import (plot_dod_comparison,
                                          plot_health_loss_comparison,
                                          plot_queue_comparison,
                                          plot_completion_rate_bar,
                                          generate_comparison_table)
        comparison_dir = os.path.join(self.base_dir, 'comparison')
        fig_dir        = os.path.join(comparison_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        plot_dod_comparison(self.result_dirs, fig_dir)
        plot_health_loss_comparison(self.result_dirs, fig_dir)
        plot_queue_comparison(self.result_dirs, fig_dir)
        summaries = {a: self.recorders[a].get_summary()
                     for a in self.recorders if self.recorders[a].get_summary()}
        if summaries:
            plot_completion_rate_bar(summaries, fig_dir)
            generate_comparison_table(summaries, comparison_dir)
