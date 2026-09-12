# DoD energy attribution: one-seed diagnostic

- Valid GitHub Actions run: <https://github.com/hhou200208-dotcom/Test1/actions/runs/34676021655>
- Source commit: `2e92be85a617f77c5a77507c74bec0f084e3049a`
- Seed: 42
- Environment: 5 x 5 satellites, lambda_high = 4.0
- Common warm-up: gold BLA-MAPPO for 5,400 slots
- Formal evaluation: 5,400 slots per candidate, continuing without reset
- CSV granularity: algorithm x seed x satellite x slot (405,000 rows)

The first attempted run (Actions `34675475097`) was rejected after its audit
showed that near a full battery the simulator curtailed solar before subtracting
same-slot load.  That ordering did not satisfy the requested net-energy balance.
Commit `2e92be8` settles solar, base energy, and task energy together before
applying the physical DoD bounds.  In the valid run, the maximum counterfactual
formula error is 0 J, the maximum actual transition residual is
`1.46e-11 J`, and no negative task-induced discharge rows remain.

## Results

| Metric | w/o DoD | w/ Linear-DoD | BLA-MAPPO |
|---|---:|---:|---:|
| Satisfaction | 0.85234 | 0.76425 | 0.84498 |
| Completion rate | 0.85242 | 0.76421 | 0.84504 |
| Completed tasks | 101,642 | 91,124 | 100,762 |
| Mean completed delay (s) | 2.5011 | 2.9417 | 2.3691 |
| Task energy / completed task (J) | 3.1094 | 1.7648 | 2.1861 |
| Lifetime loss / completed task | 2.2542e-5 | 5.5720e-6 | 7.9295e-6 |
| High-DoD energy ratio | 21.856% | 0% | 1.633% |
| High-DoD discharge ratio | 19.717% | 0% | 1.780% |
| High-DoD occupancy | 6.626% | 0% | 0.921% |
| High-DoD satellite-slot samples | 8,945 | 0 | 1,244 |
| DoD P95 | 0.66110 | 0.38843 | 0.47016 |

## Interpretation

BLA-MAPPO and w/o DoD have close service quality (0.736 percentage-point
satisfaction gap).  BLA-MAPPO reduces high-DoD energy activity strongly and has
lower lifetime loss per completed task.  However, its task energy per completed
task is also 29.7% lower than w/o DoD, outside the pre-specified 5% energy-match
condition.  This run therefore supports the mechanism qualitatively but does
not by itself isolate placement-at-DoD from total energy saving.

The Linear-DoD policy uses less energy and incurs less lifetime loss, but its
satisfaction is 8.07 percentage points below BLA-MAPPO.  It fails the 3-point
service-match condition and is not a like-for-like quality baseline in this
run.  A final paper claim requires equal training budgets and multiple seeds,
plus either an energy-matched operating point or conditional/matched analysis.
