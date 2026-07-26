# Experiment summary

## Outcome

The ordered `PLAN-paper.md` program is complete:

1. reproducible seeds and classical baselines;
2. size generalization;
3. vectorized environment stepping;
4. the reward-shaping phase diagram and goal-bonus extension;
5. the cross-paradigm RL-versus-supervised comparison.

The decision gates produce a coherent, non-cherry-picked result:

| Question | Gate | Outcome |
|---|---|---|
| Q1: Δ as reward vs loss weight | >5-point success, >0.5 opt-gap, or different size degradation | **PASS** |
| Q2: shaping phase diagram | ≥15-point graded-minus-sparse success with non-overlapping seed ranges | **HONEST NULL** |
| Q3: size generalization | >80% success at ≥8× size while another architecture collapses, or ranking flip | **PASS** |
| Q4: rigor | three seeds, deterministic seed path, classical controls, decoding/cycle metrics | **COMPLETE** |

The paper should lead with Q1, use Q3 as the second positive result, and retain
Q2 as an informative null. The most defensible thesis is:

> Exact directional information helps learned graph search, but the interface
> through which it is learned and the state distribution induced during
> training matter more than merely supplying the signal.

## Q1: cross-paradigm comparison

RL and supervised models use the same exact Dijkstra regret signal, frozen
graph instances, three training seeds, two parameter scales, and approximate
matched training compute. The MAC match is within 1.4%, and parameter counts
differ by at most 1.2%.

At the 25M compute match with sampled decoding:

| Scale | Suite | RL | Supervised | RL advantage |
|---|---|---:|---:|---:|
| small | train scale | 97.2% | 76.0% | +21.3 pp |
| small | near OOD | 95.8% | 53.3% | +42.4 pp |
| small | far OOD | 90.8% | 12.8% | **+78.0 pp** |
| large | train scale | 96.4% | 90.9% | +5.5 pp |
| large | near OOD | 92.5% | 62.7% | +29.9 pp |
| large | far OOD | 87.6% | 31.3% | **+56.2 pp** |

Greedy far-OOD success is 79.1% RL versus 4.7% supervised at the small scale,
and 69.2% versus 6.3% at the large scale. RL also makes fewer backward moves
and produces cheaper successful paths. The result is not carried by one cell:
24 matched cells exceed the success gate, 24 exceed the opt-gap gate, and six
degradation comparisons have non-overlapping three-seed ranges.

Interpretation: teacher forcing grades expert-state decisions, while RL learns
on trajectories induced by its own errors and recoveries. The experiment is
consistent with exposure bias, but does not isolate it from every optimizer and
data-distribution difference.

## Q2: shaping phase diagram

The main grid contains 54 five-million-step LSTM trainings:

- graded Δ reward, β=0 control, and binary backward penalty;
- horizons `4n`, `2n`, and `1.25n`;
- fixed \(n=8\)–16 and cold-start \(n=32\)–64;
- seeds 42, 43, and 44.

The optional extension repeats the full grid at goal bonuses 1 and 0, yielding
162 trainings overall.

The strongest cell is consistently cold-start \(n=32\)–64 with a `1.25n`
horizon:

| Goal bonus | Graded | Sparse | Difference |
|---:|---:|---:|---:|
| 10 | 83.2% | 71.7% | +11.5 pp |
| 1 | 83.0% | 70.9% | +12.2 pp |
| 0 | 82.5% | 70.0% | +12.4 pp |

All seed ranges separate in the strongest cells, but no mean difference reaches
the predeclared 15-point gate. The result is an honest null, not evidence that
shaping never matters. Directional feedback produces a repeatable 8–12 point
advantage on larger cold-start graphs under tight horizons; curricula and
generous rollout budgets make it nearly redundant on small graphs.

The binary penalty matches graded shaping in the strongest main-grid cell,
suggesting that direction matters there but exact regret magnitude is not
uniquely necessary.

## Q3: size generalization

Nine five-million-step policies—MLP, LSTM, and Transformer at three seeds—were
trained only through \(n=32\) and evaluated through \(n=768\) on shared
1,000-graph suites.

At 24 times the maximum training size:

| Controller | Sampled success | Greedy success | Successful-path opt gap |
|---|---:|---:|---:|
| random valid | 82.7% | — | 192.3 |
| MLP | 90.9% | 39.3% | 50.1 |
| LSTM | 93.6% | 48.3% | 55.3 |
| Transformer | 56.1% | 34.3% | 22.5 |

Q3 passes at \(n=384\), where MLP and LSTM remain above 90% while Transformer
falls to 62.8%. The Transformer is more efficient when it succeeds, so the
result is a success-efficiency tradeoff rather than uniform inferiority.

More than 92% of sampled \(n=768\) episodes exhibit a two-cycle for every
learned architecture. Sampling helps escape local cycles; it does not teach a
cycle-free deterministic algorithm.

## Q4: rigor and controls

The seed patch controls Python, NumPy, PyTorch, PufferLib, environment graph
generation, initialization, and action sampling. Filenames include the seed,
preventing concurrent manifest overwrite. Same-seed reproducibility checks
produce tensor-identical checkpoints and identical non-timing curves.

Classical controls show why success alone is insufficient:

- random-valid walking retains 82–88% success through \(n=768\) under `4n`,
  but path cost grows to 192 times optimal and two-cycles become universal;
- greedy minimum-edge walking has approximately optimal successful paths but
  below 6% success because it deterministically cycles.

All reported evaluations have zero invalid-action rate.

## Environment vectorization

The vectorized `GraphWalk.step` preserves byte-identical public traces for
graded and binary rewards and keeps the environment test suite green.

| Workload | Speedup |
|---|---:|
| transition kernel, resets disabled | 18.72× |
| full environment with graph resets | 2.01× |
| 500k-step LSTM/Muon PPO | 1.30× |

The kernel exceeds the expected 5–10× improvement, but exact graph generation
and Dijkstra during resets dominate end-to-end training. The paper reports the
1.30× PPO number rather than implying an 18.72× training speedup.

## Scale and integrity

The core program includes 183 learned training runs:

- 9 Q3 architecture/seed runs;
- 162 Q2 phase-diagram and bonus-ablation runs;
- 12 Q1 RL/supervised runs.

Q1 alone evaluates 180 fixed-suite conditions: 72 RL and 108 supervised. Its
three serialized suites and graph-generator implementation are hash-pinned.
Every transfer is checksum-verified before its remote stage is deleted.

The final local verification requires:

- exact artifact counts and seed-bearing filenames;
- exact suite SHA-256 values;
- all Q1 row counts and seed sets;
- `invalid_rate == 0.0` everywhere;
- the full Python test suite, Ruff, and `git diff --check`;
- no unresolved paper placeholders.

## Operational deviations

These deviations do not alter scientific configs:

1. PyPI's default PyTorch wheel selected CUDA 13, which does not support
   Pascal. The remote venv uses the official CUDA 12.6 wheel, verified by a
   compute-capability `(6, 1)` forward/backward smoke test.
2. The first LSTM/seed42 RL and small/seed42 supervised wrappers failed before
   model execution because `/usr/bin/time` was unavailable. Each traceback was
   retained and each job was retried exactly once with shell epoch timing.
3. Fleet briefly reported healthy detached jobs as dead after its inspection
   helper failed. Process trees, CPU clocks, CUDA contexts, logs, and final
   exit files were used as the source of truth; no healthy job was restarted.
4. An initial uv operation wrote a cache outside `~/q1-runs`. It was left
   untouched because cleanup authorization is restricted to the scratch tree.
5. Result aggregation exposed an unresolved `__file__` sibling-path bug. The
   path is now normalized and covered by a module-import regression test.

VoiceInk services were never restarted or modified during this work. Sunshine
was left running throughout. No git push is part of the handoff.

## Artifact map

- `paper/WORKSHOP-DRAFT.md` — workshop-style paper and appendix.
- `paper-results/Q1-cross-paradigm.md` — complete Q1 tables and gate audit.
- `paper-results/Q2-phase-diagram.md` — main grid and goal-bonus null.
- `paper-results/Q3-size-generalization.md` — full size curves and gate.
- `paper-results/Q4-seeds-baselines.md` — seed proof and classical controls.
- `paper-results/STEP-vectorization.md` — behavioral proof and benchmarks.
- `paper-results/LITERATURE.md` — positioning and primary references.
- `paper-results/*.json` — machine-readable aggregates and gate decisions.
- `paper-results/*.svg` — paper-ready Q1, Q2, and Q3 figures.
