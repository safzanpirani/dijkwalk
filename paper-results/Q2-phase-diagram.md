# Q2: shaping phase diagram

## Protocol

The main grid trains an LSTM policy for 5M agent steps in each of 54
conditions:

- three reward modes: graded β=1, β=0 ("sparse"), and binary backward penalty;
- budgets `4n`, `2n`, and `1.25n`;
- fixed n=8–16 and cold-start n=32–64, both without curriculum;
- training seeds 42, 43, and 44.

Every run uses the established Muon configuration (`lr=0.015`), 1,024
environments, and goal bonus 10. Independent training processes ran eight-way
with all OMP/BLAS thread counts capped at one. The optional bonus ablation
repeats the complete 54-run grid at goal bonuses 1 and 0, for 162 total
trainings.

Each checkpoint is evaluated with sampled decoding on 1,000 held-out graphs.
The same graph suite is reused across reward modes, budgets, bonus levels, and
training seeds within each size regime. The `1.25n` horizon follows the
environment's existing integer assignment, i.e. `floor(1.25 * n)`.

Primary commands:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/run_q2_phase_grid.py --workers 8 \
  --goal-bonus <10|1|0> --manifest viz-data/q2/grid-g<bonus>-manifest.json

.venv/bin/python scripts/evaluate_q2_phase_grid.py --workers 8 \
  --goal-bonus <10|1|0> --out <result.json>
.venv/bin/python scripts/merge_q2_bonus_ablation.py
.venv/bin/python scripts/plot_q2_phase_grid.py
.venv/bin/python scripts/export_q2_viz.py
```

Raw checkpoints, curves, per-process logs, and manifests remain in ignored
`viz-data/q2/`. The committed JSON files contain every fixed-suite result,
seed aggregate, training-tail summary, phase cell, and gate decision.

## Main result: goal bonus 10

Success is mean over the three training seeds. "Gap" is graded minus sparse.

| Start regime | Budget | Graded (%) | Sparse (%) | Gap (pp) |
| --- | ---: | ---: | ---: | ---: |
| n=8–16 fixed | 4n | 99.7 | 99.3 | +0.4 |
| n=8–16 fixed | 2n | 97.7 | 96.5 | +1.2 |
| n=8–16 fixed | 1.25n | 95.1 | 93.1 | +1.9 |
| n=32–64 cold | 4n | 97.1 | 89.3 | +7.7 |
| n=32–64 cold | 2n | 89.1 | 80.6 | +8.5 |
| n=32–64 cold | 1.25n | 83.2 | 71.7 | **+11.5** |

The trend is coherent: graph size creates most of the separation, and a tighter
budget strengthens it. The strongest cell also has cleanly separated seed
ranges—graded 82.1–85.2%, sparse 70.3–74.3%—but its 11.5-point mean gap is
below the predeclared 15-point threshold.

The binary policy reaches 83.2% in the strongest cell, equal to graded at the
reported precision. This suggests that directional feedback matters there, but
the exact magnitude of the Dijkstra regret is not uniquely responsible.

## Optional goal-bonus ablation

The strongest cell remains cold-start n=32–64 with a `1.25n` budget at every
bonus level:

| Goal bonus | Graded (%) | Sparse (%) | Binary (%) | Gap (pp) | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 83.2 | 71.7 | 83.2 | +11.5 | null |
| 1 | 83.0 | 70.9 | 82.3 | +12.2 | null |
| 0 | 82.5 | 70.0 | 82.3 | +12.4 | null |

Removing the bonus increases the largest observed gap by less than one point;
it does not reveal a hidden ≥15-point regime. The goal=0 row must not be
described as reward-free sparse RL: β=0 still pays negative edge cost and step
cost, and reaching the target terminates those future penalties. It removes the
terminal bonus, not all learning signal.

All 162 fixed-suite conditions and all recorded training tails have
`invalid_rate = 0`.

## Gate

**HONEST NULL.** The gate requires both a ≥15-point graded-minus-sparse success
gap and `min(graded seeds) > max(sparse seeds)` in at least one cell. The main
grid reaches 11.5 points; the optional bonus levels reach 12.2 and 12.4. No
level passes.

The defensible conclusion is not that shaping is redundant everywhere. It is
nearly redundant for small fixed graphs, while cold starts and tight horizons
produce a repeatable 8–12 point advantage that remains below the predeclared
effect-size threshold. Q3 passed its separate generalization gate, so the plan's
post-Q2 decision rule supports continuing to Q1 without promoting Q2 to the
headline.
