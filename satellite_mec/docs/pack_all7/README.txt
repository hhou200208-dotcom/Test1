7-method comparison figures — data + plotting code
===================================================

Scenario: LEO Walker satellite MEC, lambda_high=4.0, T_EVAL=5400 slots, 1 eval seed.
Physics params: kappa=1.5e-27, E_cap=54 kJ, V_f(V_DVFS)=2e17, DoD projection [0,0.8].
System size: simulated on 25 satellites; extensive quantities projected to the
192-satellite system by x(192/25). Satisfaction is an intensive rate (no projection).

Methods (7):
  BLA-MAPPO       proposed full method (gold, 32K training)
  w/o DoD         ablation: DoD/battery signal removed (MAPPO_NoBat, 8K)          *see note
  w/o Task Prior  ablation: task-level prior removed, BETA_TASK=0 (step_12000)     *see note
  MHSPO           baseline (Zhang TMC2024)
  LyDRL-DoD       baseline (Lyapunov + MARL, linear DoD)
  GDCO            baseline (game-theoretic distributed offloading)
  LSO             baseline (local-only, no inter-satellite offloading)

Files:
  series_all7.json      per-slot data for all 7 methods. Schema:
                          {method: {E, D, hl, sat_slot, sat_denom}}, each a list of
                          length 5400 (per slot).
                            E        = slot_system_energy (J, 25-sat sum)
                            D        = slot total end-to-end delay (s, 25-sat sum)
                            hl       = avg per-sat health loss that slot
                            sat_slot = slot satisfaction rate
                            sat_denom= done + timeout that slot (satisfaction is
                                       averaged only over slots with sat_denom>0)
  plot_4figs_all7.py    self-contained plotting script (numpy + scipy + matplotlib).
                          Run:  python plot_4figs_all7.py
  fig1_satisfaction_pdf.(png|pdf)   per-slot user satisfaction distribution
  fig2_delay_pdf.(png|pdf)          per-slot system total delay distribution
  fig3_energy_pdf.(png|pdf)         per-slot system total energy distribution
  fig4_cumulative_hl.(png|pdf)      system cumulative health loss over time

Final scalars (this run, 1 seed):
  method          satisfaction   cumHL(x192)
  BLA-MAPPO          0.838          14.0
  w/o DoD            0.852          34.3
  w/o Task Prior     0.837          11.2
  MHSPO              0.826          34.7
  LyDRL-DoD          0.862          32.7
  GDCO              (0.713)         17.1
  LSO               (0.269)          8.6

NOTES / caveats (important, please read):
  - Results are 1 eval seed (exploratory); multi-seed averaging is recommended
    before drawing final quantitative conclusions.
  - The two RL methods trained on old physics are being evaluated here under the new
    physics params -> off-distribution; for fair publishable numbers, retrain under
    these params.
  - Ablation training budgets differ from BLA-MAPPO's 32K: w/o DoD is 8K, w/o Task
    Prior is a step_12000 checkpoint. For a rigorous ablation, retrain both to 32K.
  - w/o Task Prior converges close to BLA-MAPPO (even slightly lower HL) under these
    params, i.e. the task-level prior shows little HL benefit here; do not overstate.
