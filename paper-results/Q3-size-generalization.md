# Q3: size generalization

## Protocol

All MLP, LSTM, and Transformer checkpoints were retrained for 5M agent steps
with Muon (`lr=0.015`) at training seeds 42, 43, and 44 after the Q4
reproducibility patch. Training uses the existing curriculum from n=8–16 up to
n=32.

Every policy is evaluated on the same 1,000 frozen-in-memory graphs at each
n ∈ {32, 48, 96, 192, 384, 768}, generated from suite seed 20260726. Evaluation
uses 256 environments per batch, a `4n` step budget, and both sampled and greedy
decoding. Reported spread is the sample standard deviation across the three
training seeds. Classical baselines are the Q4 3-seed results over the same size
set and graph distribution.

The n feature is clipped at 1 once n reaches the declared `N_CAP=64`; this keeps
all large-n observations inside the environment's `[0,1]` Box without changing
any training observation at n≤32. The learned policies therefore extrapolate the
local K=8 behavior rather than receiving an out-of-range size scalar.

Training command pattern (nine independent runs, eight-way locally):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/run_experiments.py --steps 5000000 \
  --num-envs 1024 --device cpu --only <run-name> --seed <42|43|44> \
  --tag q3 --out viz-data/q3
```

Evaluation command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python scripts/run_q3_generalization.py --workers 8
.venv/bin/python scripts/plot_q3.py
```

## Results

### Sampled-decoding success

Mean ± sample standard deviation across the three training seeds:

| n | Random valid (%) | MLP (%) | LSTM (%) | Transformer (%) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 87.67 ± 0.60 | 96.37 ± 1.04 | 98.30 ± 0.30 | 94.33 ± 1.25 |
| 48 | 85.83 ± 1.88 | 94.83 ± 1.33 | 96.73 ± 0.40 | 85.70 ± 0.56 |
| 96 | 84.97 ± 2.41 | 93.13 ± 1.16 | 95.83 ± 0.57 | 75.30 ± 0.66 |
| 192 | 85.43 ± 0.60 | 91.50 ± 1.81 | 94.50 ± 0.36 | 66.03 ± 0.72 |
| 384 | 81.73 ± 1.40 | 90.97 ± 1.18 | 94.00 ± 0.20 | 62.83 ± 0.91 |
| 768 | 82.67 ± 0.93 | 90.93 ± 1.94 | 93.57 ± 0.47 | 56.07 ± 2.40 |

The baseline and learned checkpoints do not share the same training-seed
semantics, so the random row is a distribution-matched floor rather than a
paired significance test.

### Sampled-decoding optimality gap

The ratio is computed only over successful episodes; lower is better.

| n | Random valid | MLP | LSTM | Transformer |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 15.89 ± 0.89 | 3.55 ± 0.24 | 3.85 ± 0.16 | 3.10 ± 0.04 |
| 48 | 22.46 ± 0.43 | 4.98 ± 0.21 | 5.24 ± 0.11 | 3.40 ± 0.06 |
| 96 | 36.17 ± 1.56 | 8.65 ± 0.21 | 9.46 ± 0.21 | 5.10 ± 0.18 |
| 192 | 61.25 ± 1.49 | 16.34 ± 1.29 | 17.66 ± 0.36 | 8.61 ± 0.28 |
| 384 | 109.00 ± 4.29 | 27.20 ± 0.59 | 30.65 ± 1.12 | 14.02 ± 0.74 |
| 768 | 192.27 ± 7.48 | 50.11 ± 3.39 | 55.32 ± 4.03 | 22.54 ± 1.47 |

The Transformer therefore has a success/efficiency tradeoff rather than a
uniformly worse policy: at n=768 it succeeds much less often, but successful
walks cost less than half the MLP/LSTM ratio.

### Decoding pathology at n=768

| Architecture | Sampled success (%) | Greedy success (%) | Sampled episodes with 2-cycle (%) |
| --- | ---: | ---: | ---: |
| MLP | 90.93 ± 1.94 | 39.27 ± 0.15 | 93.40 ± 0.56 |
| LSTM | 93.57 ± 0.47 | 48.33 ± 3.16 | 93.67 ± 0.40 |
| Transformer | 56.07 ± 2.40 | 34.27 ± 5.86 | 92.90 ± 0.72 |

Sampling does not prevent local 2-cycles; it lets the policy escape them. This
explains the 21.8–51.7 point sampled-versus-greedy success gaps without
pretending the underlying deterministic behavior is robust.

All nine 5M-step trainings reached curriculum n=32 with final rolling success
96.86–98.79%. Every one of the 108 evaluation conditions has
`invalid_rate = 0`.

## Gate

**PASS.** At n=384 (12× the maximum training size), sampled MLP and LSTM retain
90.97% and 94.00% success while Transformer has fallen to 62.83%. At n=768
(24×), MLP/LSTM remain above 90% and Transformer falls to 56.07%, below the
82.67% random-valid floor. This satisfies the predeclared criterion that one or
more architectures remain above 80% while another collapses; no ranking-flip
claim is needed.

The claim should be framed precisely: simple MLP/LSTM policies preserve
high-probability reachability far out of distribution under sampled decoding,
but path efficiency degrades sharply and deterministic decoding collapses.
