"""Paper-faithful configuration for the new BLA-MAPPO implementation."""

from core.config import Config


class NewBLAMAPPOConfig(Config):
    """Configuration matching Table 2 of ``test6.pdf`` where possible.

    The existing simulator does not implement the paper's geographic-grid traffic
    generator, so it keeps the repository's high/low Poisson arrival model. All
    BLA-MAPPO-specific optimization and physical parameters are kept explicit.
    """

    N_PLANES = 16
    N_SATS_PER_PLANE = 12
    N_SATS = N_PLANES * N_SATS_PER_PLANE

    T_TRAIN = 32_000
    T_WARMUP = 5_400
    T_EVAL = 5_400
    T_TOTAL = T_TRAIN + T_WARMUP + T_EVAL

    KAPPA = 1.5e-27
    E_CAP = 54_000.0
    DOD_MIN = 0.0
    DOD_MAX = 0.8
    BETA = 0.02

    W_DONE = 10.0
    W_TIMEOUT = 5.0
    W_REJECT = 5.0

    BLA_RHO_L = 10.0
    BLA_LAMBDA_Q = 1.0
    BLA_LAMBDA_L = 1.0
    BLA_ETA_C = 0.1
    BLA_T_NORM = 100
    BLA_B_MIN_RATIO = 0.2
    BLA_ADMISSION_TASKS = 6
    BLA_NORM_EPS = 1e-8

    def __init__(self, small_scale: bool = False):
        if small_scale:
            self.N_PLANES = 5
            self.N_SATS_PER_PLANE = 5
            self.N_SATS = 25
        super().__init__()
        # Table 2 fixes V_f instead of using Config's auto-calibrated value.
        self.V_DVFS = 2.0e17
        self.BLA_B_MIN = self.BLA_B_MIN_RATIO * self.E_CAP

    def get_state_dim(self) -> int:
        # Eq. (18): local(6) + four neighbours(7 each) + task(6).
        return 6 + self.N_NEIGHBORS * 7 + 6

    def get_critic_state_dim(self) -> int:
        # Actor observation without the task vector, plus Eq. (29)'s R^10 context.
        return (self.get_state_dim() - 6) + self.GLOBAL_SUMMARY_DIM

