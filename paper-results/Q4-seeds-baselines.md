# Q4: seeds and classical baselines

## Reproducibility patch

`scripts/run_experiments.py --seed N` now applies the same seed before policy
construction to Python, NumPy, and PyTorch; passes it to PufferLib training; and
passes it to the vector environment whose RNG is consumed by
`graphgen.random_graph`. The seed is part of every checkpoint, curve, and
manifest filename, so concurrent one-run processes no longer overwrite
`experiments.json`.

The graph-generation seed path is pinned by `test_seed_controls_graph_generation`.
Policy initialization was also checked directly: equal seeds produce
tensor-identical initial state dicts, while a different seed changes them. Two
independent 16,384-step Muon runs with seed 77 produced tensor-identical final
checkpoints and identical non-timing curve values.

## Classical baseline results

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/run_baselines.py --workers 8 \
  --out paper-results/q4-classical-baselines.json
```

Each cell is mean ± sample standard deviation over seeds 42, 43, and 44, with
1,000 complete episodes per seed and a `4n` step budget. Both controllers choose
only among the K exposed valid slots.

| Controller | n | Success (%) | Opt gap | Episodes with 2-cycle (%) |
| --- | ---: | ---: | ---: | ---: |
| Random valid | 32 | 87.67 ± 0.60 | 15.89 ± 0.89 | 89.87 ± 0.31 |
| Random valid | 48 | 85.83 ± 1.88 | 22.46 ± 0.43 | 92.83 ± 0.60 |
| Random valid | 96 | 84.97 ± 2.41 | 36.17 ± 1.56 | 96.63 ± 0.12 |
| Random valid | 192 | 85.43 ± 0.60 | 61.25 ± 1.49 | 98.17 ± 0.38 |
| Random valid | 384 | 81.73 ± 1.40 | 109.00 ± 4.29 | 99.20 ± 0.36 |
| Random valid | 768 | 82.67 ± 0.93 | 192.27 ± 7.48 | 99.60 ± 0.10 |
| Greedy min-weight | 32 | 5.57 ± 1.10 | 1.001 ± 0.001 | 94.43 ± 1.10 |
| Greedy min-weight | 48 | 4.57 ± 0.67 | 1.003 ± 0.005 | 95.43 ± 0.67 |
| Greedy min-weight | 96 | 1.87 ± 0.84 | 1.002 ± 0.004 | 98.13 ± 0.84 |
| Greedy min-weight | 192 | 1.00 ± 0.40 | 1.000 ± 0.000 | 99.00 ± 0.40 |
| Greedy min-weight | 384 | 0.27 ± 0.12 | 1.000 ± 0.000 | 99.73 ± 0.12 |
| Greedy min-weight | 768 | 0.37 ± 0.12 | 1.000 ± 0.000 | 99.63 ± 0.12 |

All 36 runs have `invalid_rate = 0`.

The honest baseline finding is that a `4n` budget is generous enough for a
uniform random valid walk to keep roughly 82–88% success through n=768, albeit
with catastrophically bad path cost and near-universal 2-cycles. Conversely,
the locally greedy min-weight rule mostly locks into deterministic 2-cycles and
almost never reaches the target. Q3 therefore needs opt-gap and 2-cycle curves,
not success alone, to distinguish learned behavior from the random floor.
