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
        self._capacity = max(int(n_max), int(n_max_final), 1)
        self.u = np.zeros(num_envs, dtype=np.int64)
        self.tgt = np.zeros(num_envs, dtype=np.int64)
        self.graph_n = np.zeros(num_envs, dtype=np.int64)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.max_steps = np.zeros(num_envs, dtype=np.int64)
        self.path_cost = np.zeros(num_envs, dtype=np.float64)
        self.backward = np.zeros(num_envs, dtype=np.int64)
        self.visited = np.zeros(
            (num_envs, self._capacity), dtype=bool)
        self.slots = np.full((num_envs, K), -1, dtype=np.int64)
        self.slot_w = np.zeros((num_envs, K), dtype=np.float64)
        self.slot_dphi = np.zeros((num_envs, K), dtype=np.float64)
        self._node_slots = np.full(
            (num_envs, self._capacity, K), -1, dtype=np.int64)
        self._node_slot_w = np.zeros(
            (num_envs, self._capacity, K), dtype=np.float64)
        self._node_slot_dphi = np.zeros(
            (num_envs, self._capacity, K), dtype=np.float64)

        # episode stats accumulated between reports
        self._ep = {k: [] for k in ('success', 'length', 'opt_gap',
                                    'backward_rate', 'invalid')}
        self._recent_success: list[float] = []    # rolling window for curriculum
        self.tick = 0

    def _reset_env(self, i, fill_observation=True):
        g = graphgen.random_graph(self.rng, self.n_min, self.n_max, self.w_max)
        self._start_graph(i, g, fill_observation=fill_observation)

    def _start_graph(self, i, g, fill_observation=True):
        """Initialize one row from a graph, including externally frozen suites."""
        self._ensure_capacity(g.n)
        self.graphs[i] = g
        self.u[i] = g.src
        self.tgt[i] = g.tgt
        self.graph_n[i] = g.n
        self.steps[i] = 0
        self.max_steps[i] = self.max_steps_factor * g.n
        self.path_cost[i] = 0.0
        self.backward[i] = 0
        self.visited[i].fill(False)
        self.visited[i][g.src] = True
        self._build_graph_tables(i, g)
        if fill_observation:
            self._fill_obs(i)

    def _ensure_capacity(self, required):
        if required <= self._capacity:
            return
        old_capacity = self._capacity
        self._capacity = required

        visited = np.zeros(
            (self.num_envs, self._capacity), dtype=bool)
        visited[:, :old_capacity] = self.visited
        self.visited = visited

        shape = (self.num_envs, self._capacity, K)
        node_slots = np.full(shape, -1, dtype=np.int64)
        node_slots[:, :old_capacity] = self._node_slots
        self._node_slots = node_slots

        node_slot_w = np.zeros(shape, dtype=np.float64)
        node_slot_w[:, :old_capacity] = self._node_slot_w
        self._node_slot_w = node_slot_w

        node_slot_dphi = np.zeros(shape, dtype=np.float64)
        node_slot_dphi[:, :old_capacity] = self._node_slot_dphi
        self._node_slot_dphi = node_slot_dphi

    def _build_graph_tables(self, i, g):
        """Precompute every node's exposed slots once per graph reset."""
        self._node_slots[i, :g.n].fill(-1)
        self._node_slot_w[i, :g.n].fill(0.0)
        self._node_slot_dphi[i, :g.n].fill(0.0)
        for u, neighbors in enumerate(g.adj):
            if len(neighbors) > K:
                neighbors = sorted(
                    neighbors, key=lambda vw: (vw[1], vw[0]))[:K]
            for slot, (v, w) in enumerate(neighbors):
                self._node_slots[i, u, slot] = v
                self._node_slot_w[i, u, slot] = w
                self._node_slot_dphi[i, u, slot] = g.phi[u] - g.phi[v]

    def _fill_obs(self, i):
        """Recompute neighbor slots of current node and write the obs row."""
        self._fill_obs_rows(np.asarray([i], dtype=np.int64))

    def _fill_obs_rows(self, rows):
        """Gather precomputed node tables into observation rows in one batch."""
        if len(rows) == 0:
            return
        rows = np.asarray(rows, dtype=np.int64)
        nodes = self.u[rows]
        slots = self._node_slots[rows, nodes]
        slot_w = self._node_slot_w[rows, nodes]
        slot_dphi = self._node_slot_dphi[rows, nodes]
        valid = slots >= 0
        safe_slots = np.maximum(slots, 0)

        observations = np.zeros(
            (len(rows), OBS_DIM), dtype=self.observations.dtype)
        slot_observations = observations[:, :K * F].reshape(-1, K, F)
        slot_observations[:, :, 0] = valid
        slot_observations[:, :, 1] = slot_w / self.w_max
        slot_observations[:, :, 2] = (
            self.visited[rows[:, None], safe_slots] & valid)
        slot_observations[:, :, 3] = (
            slots == self.tgt[rows, None]) & valid
        observations[:, K * F] = self.steps[rows] / self.max_steps[rows]
        observations[:, K * F + 1] = np.minimum(
            self.graph_n[rows] / N_CAP, 1.0)

        self.slots[rows] = slots
        self.slot_w[rows] = slot_w
        self.slot_dphi[rows] = slot_dphi
        self.observations[rows] = observations

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
            self._start_graph(i, graph, fill_observation=False)
        self._fill_obs_rows(np.arange(self.num_envs))
        return self.observations, []

    def step(self, actions):
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        rows = np.arange(self.num_envs)
        action_slots = np.asarray(actions).astype(np.int64, copy=False)
        destination = self.slots[rows, action_slots]
        selected_w = self.slot_w[rows, action_slots]
        selected_dphi = self.slot_dphi[rows, action_slots]
        valid = destination >= 0
        valid_rows = rows[valid]
        step_rewards = np.full(
            self.num_envs, -self.step_cost, dtype=np.float64)

        self.steps += 1
        n_invalid = int((~valid).sum())
        step_rewards[~valid] -= self.penalty
        if self.binary_penalty:
            step_rewards[valid] += (
                -selected_w[valid] / self.w_max
                - self.penalty * (selected_dphi[valid] < 0)
            )
        else:
            step_rewards[valid] += (
                -selected_w[valid] + self.beta * selected_dphi[valid]
            ) / self.w_max

        backward = valid & (selected_dphi < 0)
        self.backward[backward] += 1
        self.path_cost[valid] += selected_w[valid]
        self.u[valid] = destination[valid]
        self.visited[valid_rows, destination[valid]] = True

        reached_target = self.u == self.tgt
        truncated = ~reached_target & (self.steps >= self.max_steps)
        done = reached_target | truncated
        self.terminals[:] = reached_target
        self.truncations[:] = truncated
        step_rewards[reached_target] += self.r_goal
        self.rewards[:] = step_rewards

        for i in np.flatnonzero(done):
            self._log_episode(i, success=bool(reached_target[i]))
            self._reset_env(i, fill_observation=False)
        self._fill_obs_rows(rows)

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
