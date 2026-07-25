"""GraphWalk: walk a random weighted graph to a target node, PufferEnv-native.

Observation layout (float32, see constants below):
    [K slots x F features] ++ [steps/max_steps, n/n_cap]
    slot features: (valid, w/w_max, visited(neighbor), neighbor==target)
Action: Discrete(K) — pick a neighbor slot. Invalid slots must be masked in
the policy (valid flag is feature 0 of each slot); the env treats an invalid
pick as a no-op with a penalty so masking bugs are visible, not fatal.

Reward (per step, weights normalized by w_max):
    graded:  r = (-w + beta*(phi(u) - phi(v))) / w_max - step_cost
             with beta=1 this is exactly -Delta(u,v)/w_max  (Ng et al. 1999)
    binary:  r = -w/w_max - penalty*[phi(v) > phi(u)] - step_cost
    +r_goal on reaching the target; truncated at max_steps_factor*n steps.
"""
from __future__ import annotations

import gymnasium
import numpy as np

import pufferlib

from . import graphgen

K = 8            # neighbor slots
F = 4            # features per slot: valid, w_norm, visited, is_target
N_GLOBAL = 2     # steps/max_steps, n/n_cap
OBS_DIM = K * F + N_GLOBAL
N_CAP = 64       # normalizer for graph size feature


class GraphWalk(pufferlib.PufferEnv):
    def __init__(self, num_envs=1024, beta=1.0, binary_penalty=False,
                 penalty=1.0, r_goal=10.0, step_cost=0.01,
                 n_min=8, n_max=16, n_max_final=32, curriculum=True,
                 curriculum_threshold=0.9, w_max=9, max_steps_factor=4,
                 report_interval=128, render_mode=None, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(K)
        self.render_mode = render_mode
        self.num_agents = num_envs
        super().__init__(buf)

        self.num_envs = num_envs
        self.beta = beta
        self.binary_penalty = binary_penalty
        self.penalty = penalty
        self.r_goal = r_goal
        self.step_cost = step_cost
        self.n_min = n_min
        self.n_max = n_max
        self.n_max_final = n_max_final
        self.curriculum = curriculum
        self.curriculum_threshold = curriculum_threshold
        self.w_max = w_max
        self.max_steps_factor = max_steps_factor
        self.report_interval = report_interval

        self.rng = np.random.default_rng(seed)
        self.graphs: list[graphgen.GraphInstance] = [None] * num_envs
        self.u = np.zeros(num_envs, dtype=np.int64)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.max_steps = np.zeros(num_envs, dtype=np.int64)
        self.path_cost = np.zeros(num_envs, dtype=np.float64)
        self.backward = np.zeros(num_envs, dtype=np.int64)
        self.visited = [None] * num_envs          # bool array per env, size n
        self.slots = [None] * num_envs            # node id per slot, -1 = pad
        self.slot_w = [None] * num_envs           # edge weight per slot

        # episode stats accumulated between reports
        self._ep = {k: [] for k in ('success', 'length', 'opt_gap',
                                    'backward_rate', 'invalid')}
        self._recent_success: list[float] = []    # rolling window for curriculum
        self.tick = 0

    def _reset_env(self, i):
        g = graphgen.random_graph(self.rng, self.n_min, self.n_max, self.w_max)
        self._start_graph(i, g)

    def _start_graph(self, i, g):
        """Initialize one row from a graph, including externally frozen suites."""
        self.graphs[i] = g
        self.u[i] = g.src
        self.steps[i] = 0
        self.max_steps[i] = self.max_steps_factor * g.n
        self.path_cost[i] = 0.0
        self.backward[i] = 0
        self.visited[i] = np.zeros(g.n, dtype=bool)
        self.visited[i][g.src] = True
        self._fill_obs(i)

    def _fill_obs(self, i):
        """Recompute neighbor slots of current node and write the obs row."""
        g = self.graphs[i]
        u = self.u[i]
        nbrs = g.adj[u]
        if len(nbrs) > K:  # keep the K cheapest edges (deterministic)
            nbrs = sorted(nbrs, key=lambda vw: (vw[1], vw[0]))[:K]
        slots = np.full(K, -1, dtype=np.int64)
        slot_w = np.zeros(K, dtype=np.float64)
        obs = self.observations[i]
        obs[:] = 0.0
        for k, (v, w) in enumerate(nbrs):
            slots[k] = v
            slot_w[k] = w
            base = k * F
            obs[base + 0] = 1.0
            obs[base + 1] = w / self.w_max
            obs[base + 2] = float(self.visited[i][v])
            obs[base + 3] = float(v == g.tgt)
        obs[K * F + 0] = self.steps[i] / self.max_steps[i]
        obs[K * F + 1] = min(g.n / N_CAP, 1.0)
        self.slots[i] = slots
        self.slot_w[i] = slot_w

    def _log_episode(self, i, success):
        g = self.graphs[i]
        self._ep['success'].append(float(success))
        self._ep['length'].append(float(self.steps[i]))
        self._ep['backward_rate'].append(
            self.backward[i] / max(1, self.steps[i]))
        if success:
            self._ep['opt_gap'].append(self.path_cost[i] / max(g.phi[g.src], 1e-9))
        self._recent_success.append(float(success))
        if len(self._recent_success) > 2000:
            self._recent_success = self._recent_success[-2000:]
        if (self.curriculum and self.n_max < self.n_max_final
                and len(self._recent_success) >= 2000
                and np.mean(self._recent_success) > self.curriculum_threshold):
            self.n_max = min(self.n_max_final, self.n_max + 4)
            self._recent_success = []

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.tick = 0
        for i in range(self.num_envs):
            self._reset_env(i)
        return self.observations, []

    def reset_with_graphs(self, graphs):
        """Reset from an exact graph batch for matched-suite evaluation."""
        if len(graphs) != self.num_envs:
            raise ValueError(
                f'expected {self.num_envs} graphs, received {len(graphs)}')
        self.tick = 0
        for i, graph in enumerate(graphs):
            self._start_graph(i, graph)
        return self.observations, []

    def step(self, actions):
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        n_invalid = 0

        for i in range(self.num_envs):
            g = self.graphs[i]
            slot = int(actions[i])
            v = self.slots[i][slot]
            r = -self.step_cost
            self.steps[i] += 1

            if v < 0:  # invalid slot picked: no-op + penalty (mask bug signal)
                n_invalid += 1
                r -= self.penalty
            else:
                u = self.u[i]
                w = self.slot_w[i][slot]
                dphi = g.phi[u] - g.phi[v]
                if self.binary_penalty:
                    r += -w / self.w_max - self.penalty * float(dphi < 0)
                else:
                    r += (-w + self.beta * dphi) / self.w_max
                if dphi < 0:
                    self.backward[i] += 1
                self.path_cost[i] += w
                self.u[i] = v
                self.visited[i][v] = True

            if self.u[i] == g.tgt:
                r += self.r_goal
                self.terminals[i] = True
                self._log_episode(i, success=True)
                self._reset_env(i)
            elif self.steps[i] >= self.max_steps[i]:
                self.truncations[i] = True
                self._log_episode(i, success=False)
                self._reset_env(i)
            else:
                self._fill_obs(i)
            self.rewards[i] = r

        if n_invalid:
            self._ep['invalid'].append(n_invalid / self.num_envs)

        info = []
        self.tick += 1
        if self.tick % self.report_interval == 0 and self._ep['success']:
            info = [{
                'success_rate': float(np.mean(self._ep['success'])),
                'score': float(np.mean(self._ep['success'])),
                'episode_length': float(np.mean(self._ep['length'])),
                'backward_rate': float(np.mean(self._ep['backward_rate'])),
                'opt_gap': float(np.mean(self._ep['opt_gap'])) if self._ep['opt_gap'] else 0.0,
                'invalid_rate': float(np.mean(self._ep['invalid'])) if self._ep['invalid'] else 0.0,
                'n_max_cur': float(self.n_max),
            }]
            self._ep = {k: [] for k in self._ep}
        return (self.observations, self.rewards, self.terminals,
                self.truncations, info)

    def render(self):
        g = self.graphs[0]
        return f'env0: n={g.n} u={self.u[0]} tgt={g.tgt} step={self.steps[0]}'

    def close(self):
        pass
