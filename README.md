# dijkwalk — Dijkstra-in-a-Transformer via PufferLib (RL)

RL sibling of the supervised `dijkstrans` project (see `PLAN-pufferlib.md`).
An agent walks random connected weighted graphs toward a target node; reward is
potential-based shaping on φ(x) = dist(x, target): with β=1 the per-step reward
is exactly `-Δ(u,v)/w_max` — the literal "punish wrong direction" formulation.

## Layout

- `dijkwalk/graphgen.py` — copied verbatim from `../dijkstrans` (Dijkstra, φ, Δ)
- `dijkwalk/env.py` — `GraphWalk(pufferlib.PufferEnv)`, vectorized internally
  (`num_envs` walks per instance). K=8 neighbor slots, valid-flag in obs.
- `dijkwalk/policy.py` — `Policy` (MLP), `Recurrent` (LSTM wrapper),
  `Transformer` (encoder over neighbor-slot tokens). All mask invalid slots
  from the obs valid flag — no separate mask plumbing in the PPO loop.
- `train.py` — config = pufferlib `default.ini` + local `config.ini`, every key
  CLI-overridable (`--env.beta 0`, `--train.device cuda`, ...).
- `tests/test_env.py` — slot/adjacency mask correctness, r == -Δ/w_max,
  optimal-path telescoping, truncation, policy mask leak checks.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
# macOS: the prebuilt pufferlib wheel's C kernel may not match your torch;
# rebuild from source if `from pufferlib import _C` fails:
uv pip install -p .venv/bin/python --no-build-isolation --no-binary pufferlib \
    --reinstall --no-cache --no-deps pufferlib
.venv/bin/python -m pytest tests/ -q
```

## Train

```bash
.venv/bin/python train.py train                          # MLP, cpu
.venv/bin/python train.py train --rnn-name Recurrent     # + LSTM
.venv/bin/python train.py train --policy-name Transformer
# 3090: --train.device cuda --env.num-envs 4096 --wandb
./run_ablations.sh --train.device cuda --train.total-timesteps 50000000
```

Eval a checkpoint: `python train.py eval --load-model-path experiments/<id>.pt`

Metrics in the dashboard: `success_rate`, `opt_gap` (path cost / φ(src), →1.0),
`backward_rate` (fraction of φ-increasing moves), `invalid_rate` (must stay 0 —
nonzero means the action mask is broken), `n_max_cur` (curriculum graph size).

## Visualizer (`viz/`)

React + Vite + Tailwind app: animated walk replay (φ-heatmap nodes, policy
probability glow on candidate edges, per-step reward decomposition, Δ regret)
plus the ablation dashboard.

```bash
.venv/bin/python scripts/run_experiments.py --steps 5000000 --seed 42 --out viz-data
.venv/bin/python scripts/merge_experiments.py viz-data/experiments-*.json \
    --out viz-data/experiments.json
.venv/bin/python scripts/export_traces.py viz-data/shaping-beta1-seed42.pt \
    --rnn Recurrent --episodes 10 --out viz/public/data/traces-shaping-beta1.json
cp viz-data/*.json viz/public/data/   # experiments.json + per-run curves
cd viz && npm install && npm run dev
```

Trace exports **sample** from π rather than argmax — the learned policies rely
on stochasticity to escape 2-cycles (greedy ≈25% success vs sampled ≈87%).

## Notes

- Reward scale: weights normalized by `w_max`; PufferLib's advantage/return
  normalization stays on.
- Curriculum: n∈[8,16] → grows by 4 up to 32 whenever rolling success >0.9
  over 2000 episodes. Held-out eval at n∈[33,48]: pass
  `--env.n-min 33 --env.n-max 48 --env.curriculum False` to `eval`.
- Degree > K=8 nodes: env exposes the K cheapest edges (deterministic).
- Paper runs include the seed in every checkpoint, curve, and manifest filename.
  `--seed` controls policy initialization, PyTorch sampling, the environment,
  and the RNG consumed by `graphgen.random_graph`.
