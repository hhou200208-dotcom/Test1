"""
core/config.py
==============
全局超参数配置。

使用示例
--------
    cfg = Config()           # 标准配置
    cfg = AblationConfig()   # 消融实验（固定初始DoD）
    cfg = SensitivityVConfig(v_value=100.0)  # V值敏感性
    cfg = LoadTestConfig(lambda_value=0.08)  # 负载测试
"""

import math
from typing import Dict


class Config:
    """标准实验配置。所有字段均有类型注解和单位说明。"""

    # ── 星座参数 ──────────────────────────────────────────────
    N_PLANES: int = 5
    N_SATS_PER_PLANE: int = 5
    N_SATS: int = N_PLANES * N_SATS_PER_PLANE          # 25颗卫星
    ORBIT_HEIGHT: float = 550e3                         # m
    ORBIT_RADIUS: float = 6371e3 + ORBIT_HEIGHT         # m
    ORBIT_PERIOD: int = 5400                            # s（时隙数）
    LIGHT_RATIO: float = 0.65
    LIGHT_DURATION: int = int(ORBIT_PERIOD * LIGHT_RATIO)
    SHADOW_DURATION: int = ORBIT_PERIOD - LIGHT_DURATION
    SPEED_OF_LIGHT: float = 3e8                         # m/s

    # ── 时间参数 ──────────────────────────────────────────────
    TAU: float = 1.0                                    # s，时隙长度
    T_TRAIN: int = 64000                                # 时隙
    T_WARMUP: int = 5400
    T_EVAL: int = 5400
    N_EVAL_RUNS: int = 5
    T_TOTAL: int = T_TRAIN + T_WARMUP + T_EVAL

    # ── 任务参数 ──────────────────────────────────────────────
    LAMBDA_HIGH: float = 8.0                          # 高负载卫星到达率（tasks/slot）
    LAMBDA_LOW: float = 0.1                             # 低负载卫星到达率
    LAMBDA_HIGH_RATIO: float = 1 / 3
    LAMBDA: float = LAMBDA_HIGH * LAMBDA_HIGH_RATIO + LAMBDA_LOW * (1 - LAMBDA_HIGH_RATIO)
    S_MIN: float = 10e6                                 # bits
    S_MAX: float = 50e6                                 # bits
    S_AVG: float = (S_MIN + S_MAX) / 2
    H_MIN: float = 100.0                                # cycles/bit
    H_MAX: float = 300.0
    D_MAX_MIN: float = 1.0                              # s，最小截止时间
    D_MAX_MAX: float = 12.0                             # s，最大截止时间
    K_MAX: int = 3                                      # 最大转发跳数

    # ── 计算参数 ──────────────────────────────────────────────
    CPU_FREQ: float = 2e9                               # cycles/s
    MAX_DISPATCH: int = 6                               # 卫星最大并发任务数
    KAPPA: float = 1e-26                                # 能耗系数

    # ── 星间链路参数 ──────────────────────────────────────────
    B_MIN: float = 100e6                                # bits/s
    B_MAX: float = 300e6
    B_AVG: float = (B_MIN + B_MAX) / 2
    P_T: float = 0.1                                    # W，发射功率

    # ── 电池参数 ──────────────────────────────────────────────
    E_CAP: float = 10 * 3600                            # J，电池容量
    P_SOLAR_MAX: float = 30.0                           # W，最大太阳能功率
    P_HOUSEKEEPING: float = 10.0                        # W，维持卫星运行的基础功耗
                                                        # （姿控/OBC/热控等子系统，参考 NASA SOA 2020
                                                        #  及 Li et al. IEEE TSC 2024 式(4) 中 E_a(t) 项）
    DOD_MAX: float = 0.8
    DOD_MIN: float = 0.1
    DOD_INIT_LOW: float = 0.2                           # DoD初始值下界
    DOD_INIT_HIGH: float = 0.5                          # DoD初始值上界
    A_COEF: float = 0.8                                 # 电池健康损失曲线系数

    # ── Lyapunov 参数 ─────────────────────────────────────────
    V: float = 50.0                                     # 权衡参数
    ETA: float = 0.5                                    # DoD虚拟队列权重
    MU: float = 0.1                                     # 滑动平均系数
    ALPHA_BAR_INIT: float = MAX_DISPATCH / 2.0

    # ── MAPPO 参数 ────────────────────────────────────────────
    GAMMA: float = 0.99
    LAMBDA_GAE: float = 0.9
    EPSILON: float = 0.2                                # PPO clip ratio
    BETA: float = 0.01                                  # 熵正则系数
    LR_ACTOR: float = 3e-4
    LR_CRITIC: float = 1e-3
    MINIBATCH: int = 64
    EPOCH: int = 4
    K_ROLLOUT: int = 64                                 # rollout步长

    # ── 实验控制 ──────────────────────────────────────────────
    EVAL_INTERVAL: int = 15                             # 每隔多少次update做一次快速评估
    T_CALIBRATE: int = 5400                             # L_MAX校准时隙数
    DEBUG_T_TRAIN: int = 1000
    DEBUG_T_WARMUP: int = 200
    DEBUG_T_EVAL: int = 500
    DEBUG_T_CALIBRATE: int = 500
    DEBUG_N_EVAL_RUNS: int = 1

    # ── 网络结构 ──────────────────────────────────────────────
    HIDDEN_DIM: int = 256
    N_NEIGHBORS: int = 4

    # ── 随机种子 ──────────────────────────────────────────────
    SEED: int = 42
    SEED_TASK: int = SEED
    SEED_TASK_PARAM: int = SEED + 1
    SEED_LINK: int = SEED + 2
    SEED_NET: int = SEED + 3
    SEED_TRAIN: int = SEED + 4

    def __init__(self):
        self.B_BITS_MAX: float = self.B_MAX * self.TAU
        self.THETA: float = 2.0 * (self.S_MAX * self.MAX_DISPATCH
                                   + self.N_NEIGHBORS * self.B_BITS_MAX)
        comp_term = self.TAU * self.KAPPA * (self.CPU_FREQ ** 3)
        trans_term = self.P_T * self.N_NEIGHBORS * self.S_MAX / self.B_MIN
        self.DELTA_MAX: float = (comp_term + trans_term) / self.E_CAP
        self.B_F: float = 0.5 * (self.S_MAX * self.MAX_DISPATCH
                                  + self.N_NEIGHBORS * self.B_BITS_MAX) ** 2
        self.B_B: float = 0.5 * max(self.S_MAX * self.MAX_DISPATCH,
                                     self.CPU_FREQ * self.TAU) ** 2
        self.B_Z: float = 0.5 * (self.DELTA_MAX ** 2)
        self.B_TOTAL: float = self.N_SATS * (self.B_F + self.B_B + self.B_Z)
        self.Q_F_MAX: float = self.THETA
        self.L_MAX: float = self.DOD_MAX * (10 ** (self.A_COEF * (self.DOD_MAX - 1)))
        self.L_MAX_CALIBRATED: bool = False
        self.DELTA_DOD_MAX: float = self.DELTA_MAX
        self.Z_MAX: float = self.ORBIT_PERIOD * self.DELTA_MAX
        self._verify_power_balance()
        self._verify_resource_surplus()

        # 泊松到达上界（CDF ≥ 99.9%）
        lam = self.LAMBDA_HIGH
        cdf, k, _factorial = 0.0, 0, 1
        while True:
            cdf += math.exp(-lam) * (lam ** k) / _factorial
            if cdf >= 0.999:
                break
            k += 1
            _factorial *= k
        self.LAMBDA_MAX: float = float(k)
        self.THETA_NUM: float = 2 * (self.MAX_DISPATCH
                                     + (2 * self.N_NEIGHBORS + 1) * self.LAMBDA_MAX)

        a = self.A_COEF
        l_prime_max = (10 ** (a * (self.DOD_MAX - 1))) * (1.0 + a * math.log(10) * self.DOD_MAX)
        delta_dod_max_comp = (self.KAPPA * self.S_MAX * self.H_MAX
                              * (self.CPU_FREQ ** 2) / self.E_CAP)
        self.L_MAX_NEW_RAW: float = l_prime_max * delta_dod_max_comp
        self.Q_NORM: float = self.S_MAX * self.THETA_NUM

        print(f"[Config] LAMBDA_MAX={self.LAMBDA_MAX:.0f}, "
              f"THETA_NUM={self.THETA_NUM:.0f}, "
              f"L_MAX_NEW_RAW={self.L_MAX_NEW_RAW:.4e}, "
              f"Q_NORM={self.Q_NORM:.4e}")

    # ── 维度查询 ──────────────────────────────────────────────
    def get_state_dim(self) -> int:
        """Actor 输入维度：1(id) + 6(local) + N_nbr*5(neighbor) + 5(task) = 32"""
        return 1 + 6 + self.N_NEIGHBORS * 5 + 5

    def get_critic_state_dim(self) -> int:
        """Critic 输入维度：(state_dim - 5) * (1 + N_nbr) = 135"""
        return (self.get_state_dim() - 5) * (1 + self.N_NEIGHBORS)

    def get_action_dim(self) -> int:
        """动作空间大小：1(本地) + N_neighbors(转发) = 5"""
        return 1 + self.N_NEIGHBORS

    # ── 评估种子 ──────────────────────────────────────────────
    def get_eval_seeds(self, run_idx: int) -> Dict[str, int]:
        """
        生成第 run_idx 次评估的随机种子字典。

        Parameters
        ----------
        run_idx : 从0开始的run编号

        Returns
        -------
        {'task': int, 'task_param': int, 'dod_init': int}
        """
        base = self.SEED + 100 * (run_idx + 1)
        return {'task': base, 'task_param': base + 1, 'dod_init': base + 2}

    def get_quick_eval_seeds(self) -> Dict[str, int]:
        """训练过程中快速评估使用的固定种子。"""
        return {'task': self.SEED + 999, 'task_param': self.SEED + 1000,
                'dod_init': self.SEED + 1001}

    # ── 内部校验 ──────────────────────────────────────────────
    def _verify_power_balance(self):
        avg_nb = max(self.MAX_DISPATCH / 2, 1)
        avg_comp_power = self.KAPPA * (self.CPU_FREQ ** 3) / (avg_nb ** 2)
        avg_trans_power = self.P_T * self.LAMBDA * self.S_AVG / self.B_AVG
        avg_solar_power = self.P_SOLAR_MAX * self.LIGHT_RATIO
        total = avg_comp_power + avg_trans_power + self.P_HOUSEKEEPING
        print(f"[Config] 假设1验证（功耗/充电比={total/avg_solar_power:.6f}）:",
              "✓" if total <= avg_solar_power else "⚠ 不满足")

    def _verify_resource_surplus(self):
        print(f"[Config] 假设3验证（全局平均到达率={self.LAMBDA:.2f} / E_n={self.MAX_DISPATCH}）:",
              "✓" if self.LAMBDA <= self.MAX_DISPATCH else "⚠ 不满足")


class TrainConfig(Config):
    """标准训练配置（与 Config 相同，预留扩展）。"""


class AblationConfig(Config):
    """消融实验配置：固定初始DoD=0.3，消除初始化随机性。"""

    def __init__(self):
        super().__init__()
        self.DOD_INIT_LOW = 0.3
        self.DOD_INIT_HIGH = 0.3


class SensitivityVConfig(Config):
    """V值敏感性分析配置。"""

    def __init__(self, v_value: float):
        super().__init__()
        self.V = v_value


class LoadTestConfig(Config):
    """负载测试配置：自定义到达率。"""

    def __init__(self, lambda_value: float):
        super().__init__()
        self.LAMBDA = lambda_value
