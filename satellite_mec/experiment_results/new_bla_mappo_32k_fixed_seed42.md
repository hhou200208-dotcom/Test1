# new_BLA-MAPPO 32K fixed run (seed 42)

- GitHub Actions run: <https://github.com/hhou200208-dotcom/Test1/actions/runs/34476827622>
- Source commit: `eee76fc2b8f73d6cc185386aab945974230e4d03`
- Environment: 5 planes x 5 satellites (25 satellites)
- Training: 32,000 slots = 500 episodes, 64 slots per episode
- Evaluation: 5,400 warm-up slots followed by 5,400 recorded slots
- Seed: 42
- Saved model: `new_BLA-MAPPO-32K-fixed`

## Recorded metrics

| Metric | Value |
|---|---:|
| Mean training reward per slot | -3.432305992 |
| First 100-slot mean reward | -23.919669149 |
| Last 100-slot mean reward | 2.707674907 |
| Evaluation completion rate | 0.676562560 |
| Evaluation satisfaction | 0.748742867 |
| Mean completed-task delay | 4.237341533 s |
| Mean system energy per slot | 27.179446327 J |
| Cumulative lifetime loss | 0.017274176 |
| Mean lifetime loss per slot | 3.198921406e-6 |
| Last-640-slot mean HL | 7.853653118e-7 |
| Training wall time | 1,592.7 s |
| Recorded evaluation wall time | 94.5844 s |

## Plateau review

The curve reaches a broad high-reward region around episode 250. Over the final
125 episodes, the fitted slope is -0.001275 reward/episode with a 95% interval
of [-0.005287, 0.002738], and the half-window mean change is -0.426573. This is
consistent with a broad, noisy quasi-plateau.

The configured terminal test deliberately uses only the final 30 episodes. Its
slope is +0.056777 reward/episode with a 95% interval of
[+0.035605, +0.077948], and its half-window mean change is +0.756764. Therefore
the strict terminal verdict is `plateau_detected = false`: the final local
window is recovering upward and has not settled.

This is a single-seed diagnostic run. A paper-level convergence claim requires
multiple independent seeds with mean and confidence bands.
