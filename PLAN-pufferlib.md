# Plan B: Dijkstra-in-a-Transformer via PufferLib (RL formulation)

Sibling plan to the supervised `dijkstrans` project (see `~/.claude/plans/curried-wandering-stallman.md`).
Same core idea — a transformer that navigates weighted graphs *directionally toward the target* —
but learned with RL instead of shaped cross-entropy. Self-contained; fork from here.

## Why RL / PufferLib version is interesting

- The supervised variant grades teacher-forced tokens by Dijkstra regret Δ. The RL variant lets the
  policy **walk the graph itself** and learns from potential-based reward shaping — the literal
  "punish wrong direction" formulation (Ng et al. 1999 guarantees shaping with φ preserves the
  optimal policy).
- Direct comparison: same Δ signal used two ways (loss weighting vs reward). Which generalizes
  better to bigger graphs?
- PufferLib gives vectorized envs at millions of steps/sec (C-backed) — graph-walk envs are tiny,
  so throughput is bottlenecked by the policy net, which the 3090 handles fine.

## Environment design (PufferEnv)

- **State**: random connected weighted graph (reuse `dijkstrans/graphgen.py` verbatim — Dijkstra,
  φ table, Δ computation), current node `u`, target `t`, steps taken.
- **Observation** (fixed-size, no graph tokens — key difference from supervised):
  - local view: for each of K=8 neighbor slots: (neighbor id embed index, edge weight, visited flag),
    padded; plus current node id, target id, normalized step count.
  - Optionally a "GPS-free" mode: node ids only, forcing the net to build its own map memory
    (needs recurrent/transformer-over-time policy — PufferLib supports LSTM out of the box;
    transformer-over-history is the stretch goal that keeps the "transformer" in the title).
- **Action**: discrete choice over K neighbor slots (invalid slots masked — PufferLib supports
  action masking via `env.action_mask` or large negative logits).
- **Reward** (the punish-wrong-direction core):
  - per step: `r = -w(u,v) + β·(φ(u) - φ(v))` — i.e. negative edge cost plus potential-based
    shaping term. With β=1 this telescopes to optimal-policy-invariant shaping; the instantaneous
    reward is exactly `-Δ(v) ≤ 0` (0 only on shortest-path moves). Wrong-direction moves
    (φ increases) get reward `< -w` — punished proportionally to how wrong.
  - `+R_goal` (e.g. +10) on reaching target; small `-c` per step to discourage dithering;
    truncate at `4·n` steps.
- **Curriculum**: start n∈[8,16], expand to [8,32] on >90% success; eval on held-out n∈[33,48].

## Stack

- `pufferlib` (>= 2.x; pip install, check current API — it moves fast), `torch`, reuse `graphgen.py`.
- Env: pure-Python `pufferlib.PufferEnv` first (numpy obs buffers); port to the C `puffergrid`-style
  backend only if env becomes the bottleneck (unlikely vs policy forward pass).
- Training: PufferLib's bundled CleanRL-style PPO (`puffer train` / `clean_pufferl.py`) with:
  - default policy: their MLP + LSTM wrapper (baseline)
  - custom policy: small transformer encoder over neighbor-slot tokens (the "transformer" claim),
    ~4 layers / 128 dim; plugs in via PufferLib's policy registry.
- 3090 on maints: same deploy path as plan A (rsync, venv, detached run, JSONL/tensorboard logs).
  Note PufferLib's C extensions need a working MSVC build chain on Windows — if that fights back,
  run under WSL2 on maints or fall back to Modal (their docker image builds pufferlib cleanly).

## Experiments

1. **Shaping ablation** (headline): β = 0 (sparse: goal reward only) vs β = 1 (potential shaping)
   vs β = 1 with *binary* wrong-direction penalty (flat -p when φ increases) — shows graded
   directional shaping beats both sparse and binary punishment. Metrics: success rate,
   optimality gap (path cost / φ(src)), backward-step rate, sample efficiency.
2. **Policy ablation**: MLP vs LSTM vs transformer-over-neighbors.
3. **Cross-paradigm**: RL-β1 vs supervised soft_kl from plan A on identical eval graphs.

## Risks / notes

- PufferLib API churns between versions — pin the version, read their demo envs
  (`pufferlib/environments/`) before writing the env, don't trust memory of the API.
- PPO on graphs this small converges fast; most wall-clock goes to the LSTM/transformer policy.
  Batch envs ≥ 4096 to keep the 3090 busy.
- Action masking correctness is the #1 silent bug source (uniform-random over *invalid* actions
  looks like exploration). Unit-test the mask against `graphgen` adjacency.
- Reward scale: keep |r| ~ O(1) (normalize weights by w_max) or tune `vf_coef`; telescoped returns
  equal `-pathcost + β(φ(src)-0) + R_goal`, so big φ(src) inflates returns on big graphs —
  PufferLib's reward normalization handles this, leave it on.
