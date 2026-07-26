# Q1: Δ as reward versus Δ as loss weight

## Protocol

RL uses graded β=1 reward shaping with Muon (`lr=0.015`) for 5M and 25M transitions. Supervised training uses the sibling `dijkstrans` soft-KL objective for 20K steps, with checkpoints selected by the predeclared approximate-MAC compute matching. Every condition uses training seeds 42, 43, and 44.

All checkpoints are evaluated with greedy and sampled decoding on the same three serialized 1,000-graph suites. Reported spread is sample standard deviation across training seeds. Optimality gap is the successful-walk cost divided by shortest-path cost; lower is better.

## Parameter and compute matching

| Scale | RL model / params | Supervised model / params | Supervised step for 5M match | MAC ratio | Supervised step for 25M match | MAC ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | lstm / 183,081 | small / 185,192 | 200 | 1.010× | 1,000 | 1.009× |
| large | transformer / 531,202 | large / 532,152 | 1,500 | 0.987× | 7,500 | 0.986× |

The MAC estimate excludes optimizer and environment work; it is a transparent approximation, so both parameter and compute matchings are reported.

## Matched-compute results

### Small parameter scale, 5M compute match

| Suite | Decode | RL success (%) | Supervised success (%) | RL opt gap | Supervised opt gap | RL backward (%) | Supervised backward (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=16–32 | greedy | 90.23 ± 1.00 | 9.63 ± 0.90 | 2.31 ± 0.11 | 3.07 ± 1.18 | 20.75 ± 0.15 | 44.23 ± 0.58 |
| n=16–32 | sampled | 98.13 ± 0.25 | 81.40 ± 1.57 | 3.05 ± 0.06 | 14.38 ± 1.13 | 20.43 ± 0.67 | 38.78 ± 0.48 |
| n=33–48 | greedy | 86.10 ± 0.70 | 6.50 ± 0.26 | 3.41 ± 0.13 | 1.35 ± 0.14 | 26.28 ± 0.42 | 45.69 ± 0.35 |
| n=33–48 | sampled | 97.40 ± 0.20 | 72.57 ± 1.46 | 4.74 ± 0.07 | 21.57 ± 1.43 | 26.86 ± 0.08 | 41.80 ± 0.41 |
| n=64–96 | greedy | 76.57 ± 2.51 | 3.23 ± 0.21 | 5.59 ± 0.09 | 1.77 ± 0.63 | 33.94 ± 0.33 | 47.28 ± 0.18 |
| n=64–96 | sampled | 95.40 ± 0.75 | 67.77 ± 2.26 | 8.87 ± 0.08 | 36.78 ± 3.38 | 33.35 ± 0.33 | 44.14 ± 0.22 |

### Small parameter scale, 25M compute match

| Suite | Decode | RL success (%) | Supervised success (%) | RL opt gap | Supervised opt gap | RL backward (%) | Supervised backward (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=16–32 | greedy | 94.13 ± 0.95 | 18.47 ± 0.76 | 2.18 ± 0.04 | 4.70 ± 0.56 | 20.02 ± 0.38 | 42.36 ± 0.55 |
| n=16–32 | sampled | 97.23 ± 0.15 | 75.97 ± 1.04 | 2.43 ± 0.09 | 14.43 ± 0.81 | 19.88 ± 0.21 | 38.42 ± 0.64 |
| n=33–48 | greedy | 90.53 ± 0.60 | 12.23 ± 2.35 | 3.13 ± 0.06 | 5.67 ± 1.55 | 25.83 ± 0.01 | 45.16 ± 0.44 |
| n=33–48 | sampled | 95.77 ± 0.55 | 53.33 ± 1.01 | 3.70 ± 0.15 | 19.16 ± 0.76 | 25.69 ± 0.94 | 42.65 ± 0.19 |
| n=64–96 | greedy | 79.13 ± 0.85 | 4.67 ± 0.35 | 5.29 ± 0.39 | 14.14 ± 5.80 | 33.94 ± 0.77 | 46.96 ± 0.17 |
| n=64–96 | sampled | 90.80 ± 1.57 | 12.77 ± 0.58 | 6.77 ± 0.29 | 23.07 ± 1.57 | 33.41 ± 0.22 | 46.19 ± 0.17 |

### Large parameter scale, 5M compute match

| Suite | Decode | RL success (%) | Supervised success (%) | RL opt gap | Supervised opt gap | RL backward (%) | Supervised backward (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=16–32 | greedy | 87.60 ± 0.40 | 34.50 ± 12.37 | 1.96 ± 0.02 | 3.66 ± 1.43 | 20.64 ± 0.03 | 35.04 ± 5.06 |
| n=16–32 | sampled | 96.00 ± 0.26 | 82.23 ± 8.09 | 2.42 ± 0.02 | 10.19 ± 2.81 | 19.91 ± 0.12 | 33.63 ± 3.23 |
| n=33–48 | greedy | 78.73 ± 0.15 | 14.20 ± 1.64 | 2.42 ± 0.01 | 10.13 ± 6.79 | 26.66 ± 0.01 | 44.84 ± 0.24 |
| n=33–48 | sampled | 89.27 ± 0.85 | 59.90 ± 9.61 | 3.20 ± 0.16 | 20.84 ± 3.30 | 26.22 ± 0.36 | 42.36 ± 0.58 |
| n=64–96 | greedy | 67.00 ± 3.38 | 4.10 ± 0.53 | 3.53 ± 0.16 | 15.71 ± 7.29 | 33.98 ± 0.32 | 47.26 ± 0.15 |
| n=64–96 | sampled | 77.40 ± 0.36 | 24.50 ± 11.79 | 4.63 ± 0.15 | 48.62 ± 26.97 | 33.69 ± 0.15 | 46.22 ± 0.26 |

### Large parameter scale, 25M compute match

| Suite | Decode | RL success (%) | Supervised success (%) | RL opt gap | Supervised opt gap | RL backward (%) | Supervised backward (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=16–32 | greedy | 87.10 ± 0.53 | 66.93 ± 2.75 | 1.92 ± 0.03 | 2.54 ± 0.10 | 20.66 ± 0.04 | 22.95 ± 0.51 |
| n=16–32 | sampled | 96.37 ± 0.42 | 90.90 ± 1.25 | 2.52 ± 0.08 | 5.77 ± 0.33 | 20.21 ± 0.23 | 26.30 ± 0.10 |
| n=33–48 | greedy | 78.93 ± 0.32 | 22.90 ± 4.65 | 2.52 ± 0.09 | 10.17 ± 2.52 | 26.71 ± 0.01 | 42.48 ± 0.46 |
| n=33–48 | sampled | 92.53 ± 1.78 | 62.67 ± 4.17 | 3.46 ± 0.19 | 20.65 ± 0.90 | 26.18 ± 0.20 | 41.00 ± 0.53 |
| n=64–96 | greedy | 69.23 ± 0.15 | 6.27 ± 1.64 | 3.69 ± 0.11 | 48.15 ± 0.89 | 33.74 ± 0.03 | 47.19 ± 0.25 |
| n=64–96 | sampled | 87.57 ± 3.09 | 31.33 ± 6.47 | 6.04 ± 0.46 | 58.85 ± 9.20 | 33.63 ± 0.53 | 46.23 ± 0.10 |

## Supervised 20K-step reference

These checkpoints are the fixed converged reference, not a compute-matched claim.

| Scale | Suite | Decode | Success (%) | Opt gap | Backward (%) |
| --- | --- | --- | ---: | ---: | ---: |
| large | n=64–96 | greedy | 7.00 ± 1.04 | 40.10 ± 8.84 | 46.93 ± 0.17 |
| large | n=64–96 | sampled | 29.40 ± 5.12 | 47.42 ± 8.51 | 46.01 ± 0.27 |
| large | n=33–48 | greedy | 25.73 ± 4.58 | 9.95 ± 2.45 | 42.57 ± 0.27 |
| large | n=33–48 | sampled | 59.90 ± 4.52 | 21.05 ± 1.43 | 40.88 ± 0.32 |
| large | n=16–32 | greedy | 76.37 ± 1.94 | 2.35 ± 0.15 | 20.38 ± 0.76 |
| large | n=16–32 | sampled | 91.07 ± 2.73 | 5.05 ± 0.59 | 24.27 ± 0.54 |
| small | n=64–96 | greedy | 8.33 ± 2.73 | 27.85 ± 10.85 | 46.85 ± 0.55 |
| small | n=64–96 | sampled | 46.30 ± 10.10 | 52.12 ± 21.23 | 45.59 ± 0.85 |
| small | n=33–48 | greedy | 25.90 ± 0.70 | 11.79 ± 2.57 | 42.53 ± 0.18 |
| small | n=33–48 | sampled | 59.63 ± 5.86 | 18.35 ± 0.81 | 40.78 ± 0.61 |
| small | n=16–32 | greedy | 70.30 ± 2.18 | 2.25 ± 0.23 | 22.85 ± 0.67 |
| small | n=16–32 | sampled | 84.33 ± 1.90 | 4.87 ± 0.21 | 26.46 ± 0.53 |

## Gate

**PASS.** 24 matched cells exceed the strict 5-point success threshold, 24 exceed the strict 0.5 optimality-gap threshold, and 6 degradation comparisons have non-overlapping three-seed ranges.

The largest absolute success difference is 80.60 points at small/5m/train-scale/greedy. The largest absolute optimality-gap difference is 52.82 at large/25m/far-ood/sampled.

The pass/fail decision is computed directly from every matched cell; no condition was selected or rerun to improve significance.

### Train-scale to far-OOD success degradation

| Scale | Budget | Decode | RL drop (pp) | Supervised drop (pp) | Difference (pp) | Seed ranges non-overlap |
| --- | --- | --- | ---: | ---: | ---: | --- |
| small | 5m | greedy | 13.67 ± 3.02 | 6.40 ± 0.98 | -7.27 | yes |
| small | 5m | sampled | 2.73 ± 0.61 | 13.63 ± 1.89 | +10.90 | yes |
| small | 25m | greedy | 15.00 ± 1.57 | 13.80 ± 0.50 | -1.20 | no |
| small | 25m | sampled | 6.43 ± 1.59 | 63.20 ± 1.50 | +56.77 | yes |
| large | 5m | greedy | 20.60 ± 3.42 | 30.40 ± 11.84 | +9.80 | no |
| large | 5m | sampled | 18.60 ± 0.56 | 57.73 ± 4.33 | +39.13 | yes |
| large | 25m | greedy | 17.87 ± 0.64 | 60.67 ± 1.95 | +42.80 | yes |
| large | 25m | sampled | 8.80 ± 3.44 | 59.57 ± 7.54 | +50.77 | yes |

## Integrity checks

All 180 fixed-suite run rows have `invalid_rate = 0.0`. Suite hashes:

- `far-ood`: `d93e4b086a054660f3e40d0e8d7b0170abb35569bae29e6d8d7a359207d83bbc`
- `near-ood`: `dba06cb89c76e674abc8d960ace768eca1ac13db2d2111eb724f8a8847b6145c`
- `train-scale`: `c848ff46125e1cf6998a505a094a6efeb3e170091a67b7b241623a5a42727463`

Primary analysis commands:

```bash
.venv/bin/python -m scripts.merge_q1_evaluations ...
.venv/bin/python -m scripts.analyze_q1
.venv/bin/python -m scripts.report_q1
.venv/bin/python -m scripts.plot_q1
```
