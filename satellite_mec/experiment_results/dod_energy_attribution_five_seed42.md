# Five-policy DoD energy attribution (seed 42)

Cloud run: https://github.com/hhou200208-dotcom/Test1/actions/runs/34677522350

All five policies used the same seed-42 environment construction and the same
gold BLA-MAPPO trajectory for 5,400 warm-up slots. Formal evaluation continued
from that state for 5,400 slots without reset. MHSPO's DOGD predictor observed
the gold warm-up trajectory without controlling the environment. GDCO has no
learned warm-up state. The CSV contains 675,000 data rows (five policies, 25
satellites, 5,400 formal slots).

| Metric | w/o DoD | w/ Linear-DoD | BLA-MAPPO | GDCO | MHSPO |
|---|---:|---:|---:|---:|---:|
| Satisfaction | 0.852337 | 0.764245 | 0.844979 | 0.711883 | 0.821367 |
| Completed tasks | 101,642 | 91,124 | 100,762 | 84,865 | 97,948 |
| Mean completed delay (s) | 2.5011 | 2.9417 | 2.3691 | 4.1540 | 2.8973 |
| Task energy (J/completed task) | 3.1094 | 1.7648 | 2.1861 | 2.4432 | 3.2149 |
| Lifetime loss/completed task | 2.2542e-5 | 5.5720e-6 | 7.9295e-6 | 1.1349e-5 | 2.1353e-5 |
| High-DoD energy ratio | 21.856% | 0% | 1.633% | 3.790% | 26.639% |
| High-DoD discharge ratio | 19.717% | 0% | 1.780% | 4.205% | 23.949% |
| High-DoD occupancy | 6.626% | 0% | 0.921% | 1.170% | 7.415% |
| DoD P95 | 0.66110 | 0.38843 | 0.47016 | 0.46570 | 0.68609 |

The common warm-up fingerprint matched for all policies. For every algorithm,
the raw counterfactual formula error was 0 J, the maximum actual battery
transition residual was 1.46e-11 J, and negative task-induced discharge rows
were zero.

BLA-MAPPO and MHSPO satisfy the pre-registered three-percentage-point service
matching rule (2.361 percentage points apart). Relative to MHSPO, BLA-MAPPO
uses 32.0% less task energy per completion, incurs 62.9% less lifetime loss per
completion, and reduces high-DoD occupancy by 87.6%. Relative to GDCO,
BLA-MAPPO has 13.31 percentage points higher satisfaction, 10.5% less task
energy per completion, and 30.1% less lifetime loss per completion.

No baseline is within the pre-registered 5% energy-per-task matching threshold
of BLA-MAPPO. Consequently this one-seed run supports a joint energy/DoD
mechanism advantage but does not isolate a same-energy causal scheduling
effect. Cross-seed confidence intervals are not available.
