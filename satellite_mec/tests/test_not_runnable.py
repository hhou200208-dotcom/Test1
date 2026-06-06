"""
tests/test_not_runnable.py
==========================
【⚠ 不可独立运行】集成测试 / 长时间运行测试。

原因
----
以下测试由于以下条件，无法在 CI/单机短时间内直接运行：

1. 需要预训练模型文件（.pth）才能加载 MAPPO 权重
2. 训练过程需要 GPU 或数小时 CPU 时间（T_TRAIN=64000 时隙）
3. 需要完整的评估流程（5×5400 时隙）

如何使其可运行
--------------
Option A（推荐）：使用 debug 模式
    python main_run.py --debug   # T_TRAIN=1000, T_EVAL=500, N_RUNS=1

Option B：修改配置参数
    cfg = Config()
    cfg.T_TRAIN  = 500
    cfg.T_EVAL   = 200
    cfg.N_EVAL_RUNS = 1

Option C：加载已保存的模型
    policy = MAPPOPolicy(cfg)
    policy.load('./results/xxx/MAPPO/model')
    # 然后直接跑评估

标记方式
--------
所有不可运行的测试用例均以 @unittest.skip(...) 装饰，
并在 skip 消息中注明原因和解决方案。
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMAPPOTraining(unittest.TestCase):
    """
    MAPPO 训练流程测试。

    ⚠ 不可独立运行：需要 ~数小时 CPU/GPU 时间（T_TRAIN=64000）
    解决方案：pytest tests/test_not_runnable.py -k MAPPO --override_t_train=1000
    """

    @unittest.skip("需要完整训练时间（T_TRAIN=64000 时隙，约数小时），"
                   "请用 --debug 参数运行 main_run.py")
    def test_mappo_convergence(self):
        """训练后完成率应高于 LocalOnly 基线。"""
        from core import Config, SatelliteMECEnv
        from training import MAPPOPolicy
        cfg = Config()
        env = SatelliteMECEnv(cfg)
        policy = MAPPOPolicy(cfg)
        # ... 训练 + 评估逻辑

    @unittest.skip("需要完整训练时间")
    def test_mappo_no_dod_ablation(self):
        """MAPPO_NoDod 消融：关闭 DoD 惩罚后健康损失应升高。"""
        pass

    @unittest.skip("需要完整训练时间")
    def test_v_sensitivity_tradeoff(self):
        """V 值增大时：queue_backlog 应升高，health_loss 应降低。"""
        pass


class TestFullEvaluation(unittest.TestCase):
    """
    完整评估流程测试（5 run × 5400 时隙）。

    ⚠ 不可独立运行：需要预先完成训练，且评估本身耗时约 30 分钟
    """

    @unittest.skip("需要预训练模型文件（./results/xxx/MAPPO/model/）"
                   "以及 5×5400=27000 时隙的评估时间")
    def test_mappo_beats_baselines_in_cr(self):
        """MAPPO 完成率应高于所有基线（95% CI 不重叠）。"""
        pass

    @unittest.skip("需要完整评估流程")
    def test_mhspo_baseline_data_validity(self):
        """
        MHSPO Baseline 数据正确性验证。

        原始问题（已修复）：
            baseline 非常差的根因是 run_baselines_only() 中
            `for run_idx in range(1)` 只跑1次，且 T_EVAL=5400 时隙
            的统计量无法代表稳定状态（前 ~1000 时隙是暖机期）。

        修复后预期：
            N_EVAL_RUNS=5，每次独立种子，MHSPO 完成率应 > 0.5
        """
        pass


class TestModelSaveLoad(unittest.TestCase):
    """
    模型保存/加载测试。

    ⚠ 不可独立运行：需要先完成训练，才有模型文件可以加载
    """

    @unittest.skip("需要先运行训练才有 model/actor.pth 等文件")
    def test_save_and_reload(self):
        """保存后重新加载，策略行为应与保存前完全一致。"""
        pass


class TestBaselineDataCorrectness(unittest.TestCase):
    """
    Baseline 数据正确性验证（需要完整 5-run 评估）。

    记录已修复的根本原因和预期改善范围，供回归时参考。
    """

    @unittest.skip("需要完整的 5-run 评估（约 30 分钟）")
    def test_baseline_cr_not_near_zero(self):
        """
        修复前根因分析：
        ----------------
        1. run_baselines_only() 中 `range(1)` → 只跑1次，统计噪声极大
        2. 前 T_WARMUP=5400 时隙的队列填充期没有隔离，
           评估期恰好落在暂态，CR 严重偏低
        3. MetricsRecorder.record_eval_run 的 completion_rate
           用 total_done/total_arrived，但 total_done 在暂态期极小

        修复方案：
        ---------
        - N_EVAL_RUNS 恢复为 cfg.N_EVAL_RUNS (=5)，见 main_run.py
        - run_warmup() 独立于 run_evaluation()，确保评估时队列稳定

        预期结果（5-run 均值）：
        -----------------------
        LocalOnly      CR ≈ 0.45 ~ 0.60
        GreedyDelay    CR ≈ 0.55 ~ 0.70
        LyapunovGreedy CR ≈ 0.60 ~ 0.75
        MHSPO          CR ≈ 0.55 ~ 0.70
        """
        pass
