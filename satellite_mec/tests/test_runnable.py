"""
tests/test_runnable.py
======================
【可独立运行】单元测试。

运行方式
--------
    cd satellite_mec
    python -m pytest tests/test_runnable.py -v

或直接运行：
    python tests/test_runnable.py

覆盖范围
--------
- Config：维度计算、种子生成
- Task：状态机转换、时间计算、可行性判断
- LyapunovCalculator：代价函数数值、DoD增量
- Satellite：队列操作、动作掩码、状态向量形状
- Constellation：拓扑构建、任务生成
- SatelliteMECEnv：reset/step 基本流程
- BaselinePolicies：输出类型和合法性校验
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import numpy as np

from core.config import Config
from core.task import Task
from core.lyapunov import LyapunovCalculator
from core.satellite import Satellite
from core.constellation import Constellation
from core.env import SatelliteMECEnv
from baselines.deterministic import LocalOnlyPolicy, GreedyDelayPolicy, LyapunovGreedyPolicy
from baselines.mhspo import MHSPOPolicy, DOGDPredictor


# ──────────────────────────────────────────────────────────────
# 共用轻量配置（Debug模式，快速运行）
# ──────────────────────────────────────────────────────────────
def make_debug_config() -> Config:
    cfg = Config.__new__(Config)
    # 直接赋值绕过 __init__ 打印，再手动调 __init__
    return Config()


class TestConfig(unittest.TestCase):
    """Config 类的单元测试（无 IO 依赖，可独立运行）。"""

    def setUp(self):
        self.cfg = Config()

    def test_state_dim(self):
        """状态维度应为32（删除天气状态后）。"""
        self.assertEqual(self.cfg.get_state_dim(), 32)

    def test_critic_state_dim(self):
        """Critic状态维度应为135。"""
        self.assertEqual(self.cfg.get_critic_state_dim(), 135)

    def test_action_dim(self):
        """动作维度应为5（本地+4邻居）。"""
        self.assertEqual(self.cfg.get_action_dim(), 5)

    def test_eval_seeds_uniqueness(self):
        """不同run的种子应不同。"""
        s0 = self.cfg.get_eval_seeds(0)
        s1 = self.cfg.get_eval_seeds(1)
        self.assertNotEqual(s0['task'], s1['task'])
        self.assertNotEqual(s0['task_param'], s1['task_param'])

    def test_eval_seeds_keys(self):
        """种子字典应包含3个键。"""
        seeds = self.cfg.get_eval_seeds(0)
        self.assertSetEqual(set(seeds.keys()), {'task', 'task_param', 'dod_init'})

    def test_lambda_max_positive(self):
        self.assertGreater(self.cfg.LAMBDA_MAX, 0)

    def test_q_norm_positive(self):
        self.assertGreater(self.cfg.Q_NORM, 0)


class TestTask(unittest.TestCase):
    """Task 状态机的单元测试。"""

    def _make_task(self, arrive=0, deadline=10.0, size=20e6, cpu=200.0) -> Task:
        return Task(
            task_id=(0, 0, 0), size=size, cpu_cycles=cpu,
            deadline=deadline, arrive_slot=arrive, access_sat=0,
        )

    def test_initial_state(self):
        task = self._make_task()
        self.assertEqual(task.status, 'queuing')
        self.assertEqual(task.hops, 0)
        self.assertAlmostEqual(task.processed, 0.0)
        self.assertEqual(task.current_sat, task.access_sat)

    def test_remain_time_at_arrival(self):
        """到达时隙的剩余时间应等于 deadline。"""
        task = self._make_task(arrive=10, deadline=8.0)
        self.assertAlmostEqual(task.remain_time(10), 8.0)

    def test_remain_time_decrease(self):
        """经过5个时隙，剩余时间应减少5秒。"""
        task = self._make_task(arrive=0, deadline=10.0)
        self.assertAlmostEqual(task.remain_time(5), 5.0)

    def test_is_timeout_before(self):
        task = self._make_task(arrive=0, deadline=10.0)
        self.assertFalse(task.is_timeout(9))

    def test_is_timeout_at(self):
        task = self._make_task(arrive=0, deadline=10.0)
        self.assertTrue(task.is_timeout(10))

    def test_forward_accumulates_delay(self):
        task = self._make_task()
        b_nm, t_nm = 200e6, 0.005
        expected = task.size / b_nm + t_nm
        task.forward(b_nm, t_nm)
        self.assertEqual(task.hops, 1)
        self.assertAlmostEqual(task.trans_delay_acc, expected, places=6)

    def test_update_processed_partial(self):
        task = self._make_task(size=20e6)
        task.update_processed(10e6)
        self.assertAlmostEqual(task.processed, 10e6)
        self.assertEqual(task.status, 'queuing')

    def test_update_processed_complete(self):
        task = self._make_task(size=20e6)
        task.update_processed(20e6)
        self.assertTrue(task.is_done())
        self.assertEqual(task.status, 'done')

    def test_set_timeout(self):
        task = self._make_task()
        task.set_timeout()
        self.assertEqual(task.status, 'timeout')

    def test_feasible_local_when_enough_time(self):
        cfg  = Config()
        task = self._make_task(arrive=0, deadline=12.0, size=10e6, cpu=100.0)
        # nb_hat=0, 只有自己，应可完成
        feasible = task.feasible_local(nb_hat=0, cpu_freq_n=cfg.CPU_FREQ,
                                       tau=cfg.TAU, current_slot=0)
        self.assertTrue(feasible)

    def test_get_remaining_size(self):
        task = self._make_task(size=20e6)
        task.update_processed(5e6)
        self.assertAlmostEqual(task.get_remaining_size(), 15e6)


class TestLyapunov(unittest.TestCase):
    """LyapunovCalculator 数值测试。"""

    def setUp(self):
        self.cfg  = Config()
        self.calc = LyapunovCalculator(self.cfg)

    def _make_task(self) -> Task:
        return Task(task_id=(0, 0, 0), size=20e6, cpu_cycles=200.0,
                    deadline=8.0, arrive_slot=0, access_sat=0)

    def test_health_loss_positive(self):
        """健康损失函数对 dod>0 应为正值。"""
        self.assertGreater(self.calc.health_loss(0.5), 0.0)

    def test_health_loss_linear_mode(self):
        calc_lin = LyapunovCalculator(self.cfg, linear_loss=True)
        self.assertAlmostEqual(calc_lin.health_loss(0.5), 0.5)
        self.assertAlmostEqual(calc_lin.health_loss_deriv(0.5), 1.0)

    def test_delta_dod_comp_zero_nb(self):
        task = self._make_task()
        self.assertEqual(self.calc.delta_dod_comp(task, nb_next=0), 0.0)

    def test_delta_dod_comp_positive(self):
        task = self._make_task()
        ddod = self.calc.delta_dod_comp(task, nb_next=2)
        self.assertGreater(ddod, 0.0)

    def test_delta_dod_trans_zero_bnm(self):
        task = self._make_task()
        self.assertEqual(self.calc.delta_dod_trans(task, b_nm=0), 0.0)

    def test_normalized_local_cost_finite(self):
        cfg  = self.cfg
        task = self._make_task()
        sat_state = {'dod': 0.3, 'n_f': 2, 'nb_hat': 1, 'z_hat': 0.0, 'qf_size': 1e6}
        cost = self.calc.normalized_local_cost(task, sat_state, nb_next=2, current_slot=0)
        self.assertTrue(math.isfinite(cost))

    def test_normalized_forward_cost_finite(self):
        task = self._make_task()
        sat_state = {'dod': 0.3, 'n_f': 2, 'nb_hat': 1, 'z_hat': 0.0, 'qf_size': 1e6}
        nb_state  = {'qf_size': 0.5e6, 'n_f': 1}
        cost = self.calc.normalized_forward_cost(
            task, sat_state, nb_state, b_nm=200e6, current_slot=0)
        self.assertTrue(math.isfinite(cost))

    def test_no_dod_penalty_mode(self):
        """禁用 DoD 惩罚后代价应与启用时不同（通常更小）。"""
        cfg       = self.cfg
        task      = self._make_task()
        sat_state = {'dod': 0.5, 'n_f': 3, 'nb_hat': 2, 'z_hat': 0.1, 'qf_size': 2e6}
        calc_full = LyapunovCalculator(cfg, use_dod_penalty=True, use_battery_loss=True)
        calc_none = LyapunovCalculator(cfg, use_dod_penalty=False, use_battery_loss=False)
        cost_full = calc_full.normalized_local_cost(task, sat_state, nb_next=3, current_slot=0)
        cost_none = calc_none.normalized_local_cost(task, sat_state, nb_next=3, current_slot=0)
        self.assertNotEqual(cost_full, cost_none)


class TestSatellite(unittest.TestCase):
    """Satellite 状态机和向量维度测试。"""

    def setUp(self):
        self.cfg = Config()
        self.sat = Satellite(
            sat_id=0,
            neighbors=[1, 5, 20, 24],
            link_rates={1: 200e6, 5: 150e6, 20: 180e6, 24: 220e6},
            prop_delays={1: 0.001, 5: 0.002, 20: 0.003, 24: 0.001},
            solar_seq=np.ones(self.cfg.T_TOTAL, dtype=np.float32) * 15.0,
            config=self.cfg,
        )

    def _make_task(self) -> Task:
        return Task(task_id=(0, 0, 0), size=20e6, cpu_cycles=200.0,
                    deadline=8.0, arrive_slot=0, access_sat=0)

    def test_reset_clears_queues(self):
        task = self._make_task()
        self.sat.forward_queue.append(task)
        self.sat.reset()
        self.assertEqual(len(self.sat.forward_queue), 0)
        self.assertEqual(len(self.sat.compute_queue), 0)
        self.assertEqual(self.sat.nb, 0)

    def test_admit_task_succeeds(self):
        task = self._make_task()
        result = self.sat.admit_task(task)
        self.assertTrue(result)
        self.assertEqual(len(self.sat.forward_queue), 1)

    def test_state_vector_shape(self):
        """get_state() 应返回 (32,) 向量。"""
        task  = self._make_task()
        state = self.sat.get_state(task, current_slot=0, neighbor_info={})
        self.assertEqual(state.shape, (self.cfg.get_state_dim(),))
        self.assertEqual(state.shape[0], 32)

    def test_critic_state_shape(self):
        """get_critic_state() 应返回 (135,) 向量。"""
        cs = self.sat.get_critic_state(neighbor_info={}, current_slot=0)
        self.assertEqual(cs.shape, (self.cfg.get_critic_state_dim(),))
        self.assertEqual(cs.shape[0], 135)

    def test_action_mask_shape(self):
        mask = self.sat.get_action_mask(self._make_task(), current_slot=0, neighbor_nb={})
        self.assertEqual(mask.shape, (self.cfg.get_action_dim(),))
        self.assertEqual(mask.shape[0], 5)

    def test_action_mask_local_allowed_when_idle(self):
        """空闲卫星（nb_hat=0）的本地动作应被允许。"""
        self.sat.init_temp_state()
        mask = self.sat.get_action_mask(self._make_task(), current_slot=0,
                                        neighbor_nb={1: 0, 5: 0, 20: 0, 24: 0})
        self.assertEqual(mask[0], 1.0)

    def test_action_mask_local_blocked_at_capacity(self):
        """nb_hat >= MAX_DISPATCH 时本地动作应被屏蔽。"""
        self.sat.init_temp_state()
        self.sat.nb_hat = self.cfg.MAX_DISPATCH
        mask = self.sat.get_action_mask(self._make_task(), current_slot=0, neighbor_nb={})
        self.assertEqual(mask[0], 0.0)

    def test_update_solar(self):
        self.sat.update_solar(0)
        self.assertGreater(self.sat.solar_power, 0.0)
        self.assertEqual(self.sat.xi, 1)

    def test_remove_timeout_tasks(self):
        task = self._make_task()                     # deadline=8.0, arrive=0
        self.sat.admit_task(task)
        removed = self.sat.remove_timeout_tasks(current_slot=10)  # 已超时
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].status, 'timeout')
        self.assertEqual(len(self.sat.forward_queue), 0)

    def test_process_tasks_empty(self):
        done, timeout = self.sat.process_tasks(current_slot=0)
        self.assertEqual(done, [])
        self.assertEqual(timeout, [])

    def test_get_info_keys(self):
        info = self.sat.get_info()
        for key in ['qf_size', 'n_f', 'nb', 'dod', 'xi', 'tau_switch']:
            self.assertIn(key, info)


class TestConstellation(unittest.TestCase):
    """Constellation 构建和任务生成测试。"""

    def setUp(self):
        self.cfg   = Config()
        self.const = Constellation(self.cfg)

    def test_satellite_count(self):
        self.assertEqual(len(self.const.satellites), self.cfg.N_SATS)

    def test_neighbor_count(self):
        """每颗卫星应有恰好4个邻居。"""
        for sat in self.const.satellites:
            self.assertEqual(len(sat.neighbors), 4)

    def test_link_rates_in_range(self):
        """链路速率应在 [B_MIN, B_MAX] 范围内。"""
        for sat in self.const.satellites:
            for rate in sat.link_rates.values():
                self.assertGreaterEqual(rate, self.cfg.B_MIN)
                self.assertLessEqual(rate, self.cfg.B_MAX)

    def test_generate_tasks_keys(self):
        """generate_tasks 应为每颗卫星返回键。"""
        tasks = self.const.generate_tasks(t=0)
        self.assertEqual(set(tasks.keys()), set(range(self.cfg.N_SATS)))

    def test_generate_tasks_size_range(self):
        """生成任务的 size 应在 [S_MIN, S_MAX]。"""
        tasks = self.const.generate_tasks(t=0)
        for sat_tasks in tasks.values():
            for task in sat_tasks:
                self.assertGreaterEqual(task.size, self.cfg.S_MIN)
                self.assertLessEqual(task.size, self.cfg.S_MAX)

    def test_exchange_info_keys(self):
        info = self.const.exchange_info()
        self.assertEqual(set(info.keys()), set(range(self.cfg.N_SATS)))
        for sat_info in info.values():
            self.assertIn('nb', sat_info)
            self.assertIn('dod', sat_info)

    def test_reset_determinism(self):
        """相同种子 reset 后，生成的任务数应相同。"""
        seeds = {'task': 42, 'task_param': 43, 'dod_init': 44}
        self.const.reset(seeds=seeds)
        n1 = sum(len(v) for v in self.const.generate_tasks(0).values())
        self.const.reset(seeds=seeds)
        n2 = sum(len(v) for v in self.const.generate_tasks(0).values())
        self.assertEqual(n1, n2)

    def test_get_stats_keys(self):
        stats = self.const.get_stats()
        for key in ['avg_dod', 'max_dod', 'avg_qf_size', 'avg_health_loss']:
            self.assertIn(key, stats)


class TestSatelliteMECEnv(unittest.TestCase):
    """SatelliteMECEnv 基本流程测试（轻量，仅运行少量时隙）。"""

    def setUp(self):
        self.cfg = Config()
        self.env = SatelliteMECEnv(self.cfg)

    def test_reset_returns_dict(self):
        result = self.env.reset(phase='eval')
        self.assertIsInstance(result, dict)

    def test_step_returns_correct_types(self):
        self.env.reset(phase='eval')
        next_obs, rewards, done, info = self.env.step({})
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_info_keys(self):
        self.env.reset(phase='eval')
        _, _, _, info = self.env.step({})
        for key in ['slot', 'arrived', 'done_tasks', 'completion_rate',
                    'avg_dod', 'slot_e2e_delays', 'slot_satisfaction_rate']:
            self.assertIn(key, info)

    def test_completion_rate_in_range(self):
        self.env.reset(phase='eval')
        for _ in range(10):
            _, _, _, info = self.env.step({})
        cr = info['completion_rate']
        self.assertGreaterEqual(cr, 0.0)
        self.assertLessEqual(cr, 1.0)

    def test_get_observations_shape(self):
        self.env.reset(phase='eval')
        self.env.step({})   # 产生一些队列任务
        obs = self.env.get_observations()
        for n, sat_obs in obs.items():
            for state in sat_obs:
                self.assertEqual(state.shape, (self.cfg.get_state_dim(),))

    def test_get_critic_obs_shape(self):
        self.env.reset(phase='eval')
        critic_obs = self.env.get_critic_obs()
        for n, cs in critic_obs.items():
            self.assertEqual(cs.shape, (self.cfg.get_critic_state_dim(),))

    def test_eval_phase_extra_keys(self):
        self.env.reset(phase='eval')
        _, _, _, info = self.env.step({})
        self.assertIn('eval_completion_rate', info)
        self.assertIn('eval_satisfaction_rate', info)

    def test_determinism_with_same_seeds(self):
        """相同种子 reset 后，5个时隙的 arrived 数应相同。"""
        seeds = {'task': 42, 'task_param': 43, 'dod_init': 44}
        arrivals_1, arrivals_2 = [], []

        self.env.reset(phase='eval', seeds=seeds)
        for _ in range(5):
            _, _, _, info = self.env.step({})
            arrivals_1.append(info['arrived'])

        self.env.reset(phase='eval', seeds=seeds)
        for _ in range(5):
            _, _, _, info = self.env.step({})
            arrivals_2.append(info['arrived'])

        self.assertEqual(arrivals_1, arrivals_2)


class TestBaselinePolicies(unittest.TestCase):
    """基线策略的输出合法性测试（少量时隙，快速）。"""

    def setUp(self):
        self.cfg = Config()
        self.env = SatelliteMECEnv(self.cfg)
        self.env.reset(phase='eval', seeds={'task': 42, 'task_param': 43, 'dod_init': 44})
        # 运行几步产生队列任务
        for _ in range(3):
            self.env.step({})

    def _check_policy(self, policy):
        """通用测试：策略返回的动作应为合法整数。"""
        obs   = self.env.get_observations()
        masks = self.env.get_action_masks()
        actions = policy.get_actions(obs, masks)
        self.assertIsInstance(actions, dict)
        for n, sat_actions in actions.items():
            sat = self.env.constellation.satellites[n]
            self.assertEqual(len(sat_actions), len(masks.get(n, [])))
            for action, mask in zip(sat_actions, masks.get(n, [])):
                # 动作必须在有效范围内
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, self.cfg.get_action_dim())
                # 不应选择被屏蔽的动作（若有合法动作）
                if mask.sum() > 0:
                    self.assertEqual(mask[action], 1.0,
                                     msg=f"Policy {policy.name} chose masked action {action}")

    def test_local_only_policy(self):
        policy = LocalOnlyPolicy(self.cfg, self.env)
        self._check_policy(policy)

    def test_greedy_delay_policy(self):
        policy = GreedyDelayPolicy(self.cfg, self.env)
        self._check_policy(policy)

    def test_lyapunov_greedy_policy(self):
        # LyapunovGreedy 需要 init_temp_state
        for sat in self.env.constellation.satellites:
            sat.init_temp_state()
        policy = LyapunovGreedyPolicy(self.cfg, self.env)
        self._check_policy(policy)

    def test_mhspo_policy(self):
        policy = MHSPOPolicy(self.cfg, self.env)
        self._check_policy(policy)

    def test_policy_step_integration(self):
        """所有基线策略能完整跑5个时隙而不抛出异常。"""
        policies = [
            LocalOnlyPolicy(self.cfg, self.env),
            GreedyDelayPolicy(self.cfg, self.env),
            LyapunovGreedyPolicy(self.cfg, self.env),
            MHSPOPolicy(self.cfg, self.env),
        ]
        for policy in policies:
            self.env.reset(phase='eval', seeds={'task': 99, 'task_param': 100, 'dod_init': 101})
            for _ in range(5):
                try:
                    _, _, _, info = self.env.step(policy=policy)
                except Exception as e:
                    self.fail(f"Policy {policy.name} raised {e}")


class TestDOGDPredictor(unittest.TestCase):
    """DOGDPredictor 数值正确性测试。"""

    def setUp(self):
        self.cfg  = Config()
        self.pred = DOGDPredictor(self.cfg, sat_id=0)

    def test_initial_get_returns_fallback(self):
        val = self.pred.get(tau=0, fallback=3.0)
        self.assertGreaterEqual(val, 0.0)

    def test_update_does_not_raise(self):
        try:
            for t in range(20):
                self.pred.update(current_slot=t, actual_nb=t % 4)
        except Exception as e:
            self.fail(f"DOGDPredictor.update raised: {e}")

    def test_predictions_in_range(self):
        """所有预测值应在 [0, qb_max] 范围内。"""
        for t in range(30):
            self.pred.update(current_slot=t, actual_nb=t % 5)
        for tau, val in self.pred.predictions.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, self.pred.qb_max + 1e-6)

    def test_old_predictions_cleaned(self):
        """过期预测应被清理，predictions 中不应存在早于 current_slot 的键。"""
        for t in range(50):
            self.pred.update(current_slot=t, actual_nb=2)
        for tau in self.pred.predictions:
            self.assertGreaterEqual(tau, 49)   # 最后一次 update 的 current_slot


if __name__ == '__main__':
    unittest.main(verbosity=2)
