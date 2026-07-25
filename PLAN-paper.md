# Plan: from dijkwalk demo → workshop-paper evidence

Target: a NeurIPS/ICLR workshop submission (algorithmic reasoning or RL track),
~4 pages + appendix. Working title: *"When does directional reward shaping
matter? Same regret signal, two paradigms, one graph."*

The current repo (muon grid, 2026-07) is the baseline harness. Every experiment
below reuses `scripts/run_experiments.py` + `graphgen.py` unchanged unless noted.
Decision gates are explicit — if a gate fails, the corresponding claim is dropped
rather than massaged.

---

## Q1 — Cross-paradigm: Δ as loss weight vs Δ as reward (headline candidate)

The supervised sibling (`../dijkstrans`) grades teacher-forced tokens by Dijkstra
regret Δ; the RL version pays −Δ/w_max as reward. Same signal, two uses. Nobody
has compared these cleanly on identical graphs.

**Setup**
- Freeze one eval suite: 3 × 1,000 held-out graphs at n∈{16–32, 33–48, 64–96},
  fixed seeds, serialized to `eval-suites/*.json` and committed. Both projects
  consume this exact file.
- RL side: β=1 LSTM + Transformer at 5M and 25M steps.
- Supervised side: dijkstrans soft_kl best config, matched param counts
  (~183k and ~531k) and matched *compute* (report both matchings; reviewers ask).
- Metrics on the shared suite: success rate, opt gap, backward-step rate; for
  supervised, decode by sampling AND greedy (we know greedy≠sampled for RL —
  check whether supervised has the same gap; that contrast is itself a finding).

**Gate**: interesting if either paradigm wins by >5pts success or >0.5 opt gap
anywhere, or if their *degradation slopes* with n differ. If they're identical
everywhere, Q1 demotes to an appendix table and Q2 becomes the headline.

**Cost**: eval infra ~1 day; runs are hours on the Mac/server-pc.

## Q2 — Find the regime where shaping earns its keep

At n≤32 with a 4n step budget, sparse ≈ graded (our null). Shaping should
separate when random/undirected exploration stops reaching the goal. Sweep the
two knobs that control that:

**Grid** (LSTM policy, 3 shaping modes × grid, 5M steps each):
- step budget: 4n → 2n → 1.25n (tight budgets kill dithering strategies)
- graph size at train time: n∈[8,16] fixed (no curriculum) vs n∈[32,64] cold
  start — sparse reward should crater on big cold-start graphs, shaping shouldn't
- optional third knob if signal is weak: goal bonus 10 → 1 → 0 (with shaping,
  the telescoped return alone still points home; sparse dies without the bonus)

That's 3×3×2 ≈ 18 runs × 3 seeds = 54 runs. At ~10 min/run (LSTM, muon) this is
~2 CPU-days serial → run 8-way parallel locally or on server-pc: ~3 hours.

**Deliverable**: a phase diagram — x = step budget, y = start size, cell = success
gap (graded − sparse). One figure that answers "when does shaping matter."

**Gate**: need ≥15pt success gap somewhere with non-overlapping seed spread,
else the honest paper is "shaping is redundant given curricula + goal bonus,"
which is publishable only with Q3's generalization story attached.

## Q3 — Size generalization (the NAR-crowd question)

Train n≤32, evaluate n ∈ {48, 96, 192, 384, 768}. Local K=8 observations mean
input dim never changes — nothing about the architecture blocks this, only
whether learned behavior transfers.

**Work needed**
- `graphgen.random_graph` at n=768: Dijkstra is fine; the O(n²) spring layout in
  the *exporter* is the only quadratic bit — skip layouts for eval (viz only
  needs a few showcase graphs).
- Eval batching: current eval loops Python per env; vectorize with
  `num_envs=256` env instances as-is (already supported) — fine.
- Compare MLP / LSTM / Transformer degradation curves + a classical baseline
  row: greedy-by-weight heuristic and random-valid-walk, so the plot has floors.
- Report both sampled and greedy decoding; the stochasticity crutch may explain
  transfer differences.

**Gate**: interesting if any architecture holds >80% success at 8× training size
while others collapse, or if the ranking flips with n. Also the cheapest
experiment here — do this FIRST; its curves make every other section stronger.

## Q4 — Rigor pass (non-negotiable for submission)

- **Seeds**: 3 minimum (5 preferred) for every reported number; mean ± std bands
  in all curve plots. `run_experiments.py` needs a `--seed` that threads into
  env, torch, and graphgen (currently fixed 42) — small patch.
- **Baselines beyond ours**: (a) uniform random over valid slots, (b) greedy
  min-weight edge, (c) greedy min-(w + heuristic) if we add coordinates — cheap
  rows that anchor every table.
- **The stochasticity finding**: measure properly — success vs decoding
  temperature τ ∈ {0 (greedy), 0.5, 1, 2} for each checkpoint; count 2-cycles
  explicitly (u→v→u within 2 steps). One figure. Cite it as an observed
  pathology of local-observation graph policies, not a headline.
- **Ablate the crutches**: no-curriculum and no-goal-bonus rows so the "sparse
  is fine" claim can't be attributed to curriculum alone.
- **Literature positioning** (write early, it shapes experiments): Ng et al.
  1999 (shaping invariance), CLRS-30 / Veličković et al. (algorithmic
  reasoning, Dijkstra execution), learned heuristics for search (e.g. NeuroSAT-
  style, learned A*), reward-density studies. One pass collecting ~15 refs
  before running Q2 so the grid answers questions reviewers actually have.

## Infra

- **Where**: everything CPU-feasible except the Q3 transformer evals at n≥384
  and any 25M-step Q1 runs — put those on server-pc (CachyOS, GTX 1080):
  clean pufferlib build on Linux, `--train.device cuda`. Deploy = rsync repo +
  `uv venv` + pip install; ~30 min once.
- **Env speed**: 54-run grids make the Python env loop the bottleneck →
  numpy-vectorize `GraphWalk.step` (batch the per-env loop; ~1 focused day,
  expect 5–10×). Do this before Q2's grid, not before Q3.
- **Tracking**: keep JSON logging; add `--tag`/seed into filenames; the viz app
  already renders any experiments.json — extend it with a phase-diagram panel
  for Q2 (one new component).

## Order of operations

1. Q4 seeds patch + baselines (½ day) — everything downstream needs it.
2. Q3 size generalization (1 day incl. plots) — cheapest, informs framing.
3. Vectorize env step (1 day).
4. Q2 phase-diagram grid (1 day wall-clock with parallelism).
5. Q1 cross-paradigm (2 days incl. dijkstrans-side matching).
6. Write-up (3–4 days), viz app doubles as the figure factory.

~2 focused weeks. Decision point after step 4: if neither Q2 nor Q3 gates pass,
stop at a strong blog post + repo instead of forcing a submission.
