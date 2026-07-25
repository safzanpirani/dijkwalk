# Literature positioning for the paper experiments

This pass was completed before Q2 so the phase diagram tests reviewer-facing
questions rather than only extending the existing ablation.

## Positioning

The closest theoretical anchor is potential-based reward shaping (PBRS), which
characterizes reward transformations that preserve optimal policies while
potentially reducing learning time [1]. The empirical question here is narrower
and not answered by the invariance result: **when does an exact shortest-path
potential difference improve finite-budget policy learning, and when do a goal
bonus, curriculum, and a loose rollout budget make it redundant?** Work on
learning whether/how strongly to use shaping rewards [2, 3] reinforces the need
for a regime map rather than a single favorable setting.

The closest algorithmic-reasoning work is CLRS and its step-supervised
predecessors [5–8]. They emphasize out-of-distribution size generalization and
teacher-forced execution traces. GNARL independently reframes neural
algorithmic reasoning as RL [10], but does not make this repo's controlled
comparison: the same per-decision Dijkstra regret used as a supervised loss
weight versus an RL reward on identical graph instances.

Learned search and solver work [11–15] provides the right baseline norm:
distinguish task completion from search/path efficiency, retain classical
controllers, evaluate beyond the training size, and report decoding dynamics.
That directly motivates success, optimality gap, backward-step rate, and
2-cycle rate together.

## Consequences for Q2 and Q3

- Q2 must sweep both exploration difficulty knobs already in the plan:
  `max_steps_factor` and cold-start graph size. The Q4 baseline shows why:
  uniform random remains above 80% success under `4n`, so that cell is not a
  meaningful sparse-reward stress test.
- Goal bonus and curriculum are confounders, not cleanup ablations. If the
  planned 18-cell grid has no ≥15-point gap, run the predeclared goal-bonus
  extension rather than retuning optimizer or learning rate.
- Report learning efficiency and final performance separately. PBRS theory is
  about optimal-policy preservation, not a guarantee that PPO learns faster.
- Report path quality even when success is saturated. A random walk can solve
  these sparse connected graphs under a loose budget while being hundreds of
  times more expensive than the shortest path.
- Use matched graph suites and both greedy and sampled decoding. A high sampled
  success rate can conceal a deterministic local cycle pathology.
- Avoid claiming exact discounted-policy invariance without qualification:
  the environment uses the undiscounted potential difference
  `phi(u) - phi(v)` so β=1 equals `-Delta/w_max`, while PPO uses `gamma=0.995`.
  The Δ identity is exact; the textbook `gamma*Phi(s') - Phi(s)` PBRS form for
  the optimizer's discount is not.

## Primary references

1. Ng, Harada, and Russell (1999), [Policy invariance under reward
   transformations: Theory and application to reward
   shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
2. Hu et al. (2020), [Learning to Utilize Shaping Rewards: A New Approach of
   Reward Shaping](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b710915795b9e9c02cf10d6d2bdb688c-Abstract.html).
3. Gupta et al. (2023), [Behavior Alignment via Reward Function
   Optimization](https://openreview.net/forum?id=PfpAQuyZCB).
4. Andrychowicz et al. (2017), [Hindsight Experience
   Replay](https://proceedings.neurips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html).
5. Veličković et al. (2022), [The CLRS Algorithmic Reasoning
   Benchmark](https://proceedings.mlr.press/v162/velickovic22a.html).
6. Veličković et al. (2020), [Neural Execution of Graph
   Algorithms](https://openreview.net/forum?id=SkgKO0EtvS).
7. Ibarz et al. (2022), [A Generalist Neural Algorithmic
   Learner](https://proceedings.mlr.press/v198/ibarz22a.html).
8. Bevilacqua et al. (2023), [Neural Algorithmic Reasoning with Causal
   Regularisation](https://openreview.net/forum?id=kP2p67F4G7).
9. Dudzik and Veličković (2022), [Graph Neural Networks are Dynamic
   Programmers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8248b1ded388fcdbbd121bcdfea3068c-Abstract-Conference.html).
10. Schutz et al. (2025), [Tackling GNARLy Problems: Graph Neural Algorithmic
    Reasoning Reimagined through Reinforcement
    Learning](https://arxiv.org/abs/2509.18930).
11. Yonetani et al. (2021), [Path Planning using Neural A*
    Search](https://proceedings.mlr.press/v139/yonetani21a.html).
12. Yolcu and Poczos (2019), [Learning Local Search Heuristics for Boolean
    Satisfiability](https://proceedings.neurips.cc/paper_files/paper/2019/hash/12e59a33dea1bf0630f46edfe13d6ea2-Abstract.html).
13. Selsam et al. (2019), [Learning a SAT Solver from Single-Bit
    Supervision](https://openreview.net/forum?id=HJMC_iA5tm).
14. Khalil et al. (2017), [Learning Combinatorial Optimization Algorithms over
    Graphs](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html).
15. Bieber et al. (2020), [Learning to Execute Programs with Instruction
    Pointer Attention Graph Neural
    Networks](https://proceedings.neurips.cc/paper/2020/hash/62326dc7c4f7b849d6f013ba46489d6c-Abstract.html).
