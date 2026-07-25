# GraphWalk.step vectorization

## Change

`GraphWalk` now precomputes each graph node's exposed K-cheapest neighbor table
once at reset, stores visited state in a padded batch array, and uses NumPy
gathers for selected destinations, weights, Δφ, rewards, movement, and
observation construction. The per-step Python loop now runs only over rows whose
episodes ended, preserving graph RNG order, curriculum updates, episode logging,
and reset order.

Graph generation itself is unchanged. The slot order is unchanged for degree≤K,
and degree>K still exposes exactly the K cheapest edges with the existing
weight/node-id tiebreak.

## Behavioral proof

Before the rewrite, the scalar implementation's complete public outputs were
hashed for two 120-step traces: graded β=0.7 and binary shaping. The traces use
17 parallel environments, random valid and invalid slot indices, a one-n
horizon, frequent reporting, successes, truncations, and resets.

After vectorization, both SHA-256 digests remain byte-identical:

- graded: `d4478d0c7c04e1aa4b23dd5d487f0217a14007135da9486b328d8f48e760bf78`
- binary: `1de333b9c3e71ec61949c6c7ceb6646f5e70351da1a5b1a19d0005726694371c`

The characterization is committed in `tests/test_env.py`. The full environment
suite passes: 10 tests, including K-cheapest exposure, `r=-Δ/w_max`, optimal
telescoping, invalid actions, terminal/truncation behavior, policy masking,
large-n bounds, and fixed-suite evaluation.

## Benchmark

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/benchmark_env.py
```

Darwin arm64, Python 3.12.12, PyTorch 2.12.0. The scalar comparison is loaded
from Q3 commit `51e3715`; both versions run `num_envs=1024`, n∈[32,64], and
identical seeds. Environment figures are mean ± sample standard deviation over
three seeds.

| Workload | Scalar SPS | Vectorized SPS | Speedup |
| --- | ---: | ---: | ---: |
| Transition + observations, resets disabled | 267,751 ± 1,535 | 5,013,644 ± 95,047 | **18.72×** |
| Full random-valid environment with graph resets | 157,686 ± 1,273 | 316,578 ± 6,150 | **2.01×** |
| 500k-step cold-start LSTM/Muon PPO | 73,867 | 96,141 | **1.30×** |

The honest result is that the requested step kernel exceeds the expected 5–10×
speedup, but total training does not. After vectorization, profiling the full
environment attributes about 74% of time to exact random graph generation and
Dijkstra during resets. That work deliberately remains unchanged to preserve
the graph distribution and seed stream. Q2 wall-clock estimates should use the
measured 1.30× PPO improvement, not the 18.72× kernel number.
