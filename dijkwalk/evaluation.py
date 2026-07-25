"""Batched evaluation helpers shared by paper experiments."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .env import GraphWalk, K

ActionFn = Callable[[np.ndarray, GraphWalk], np.ndarray]
DoneFn = Callable[[np.ndarray], None]


def baseline_actions(name: str, rng: np.random.Generator) -> ActionFn:
    """Build a controller over the K exposed neighbor slots."""
    if name not in {'random-valid', 'greedy-weight'}:
        raise ValueError(f'unknown baseline: {name}')

    def choose(_observations: np.ndarray, env: GraphWalk) -> np.ndarray:
        slots = np.stack(env.slots)
        valid = slots >= 0
        if name == 'random-valid':
            counts = valid.sum(axis=1)
            return (rng.random(env.num_envs) * counts).astype(np.int64)

        weights = np.stack(env.slot_w)
        return np.where(valid, weights, np.inf).argmin(axis=1)

    return choose


def evaluate_episodes(
    env: GraphWalk,
    choose_actions: ActionFn,
    episodes: int,
    seed: int,
    reset_done: DoneFn | None = None,
) -> dict[str, float | int]:
    """Evaluate complete episodes while the env auto-resets finished rows."""
    observations, _ = env.reset(seed=seed)
    completed: dict[str, list[float]] = {
        'success': [],
        'length': [],
        'opt_gap': [],
        'backward_rate': [],
        'invalid_rate': [],
        'two_cycle_rate': [],
        'has_two_cycle': [],
    }
    previous_u = np.full(env.num_envs, -1, dtype=np.int64)
    two_cycles = np.zeros(env.num_envs, dtype=np.int64)
    invalid_moves = np.zeros(env.num_envs, dtype=np.int64)

    while len(completed['success']) < episodes:
        actions = np.asarray(
            choose_actions(observations, env), dtype=np.int64)
        if actions.shape != (env.num_envs,):
            raise ValueError(
                f'actions must have shape {(env.num_envs,)}, got {actions.shape}')
        if np.any((actions < 0) | (actions >= K)):
            raise ValueError('actions must be valid slot indices')

        rows = np.arange(env.num_envs)
        slots = np.stack(env.slots)
        weights = np.stack(env.slot_w)
        current_u = env.u.copy()
        destination = slots[rows, actions]
        valid = destination >= 0
        invalid_moves += ~valid
        two_cycles += valid & (destination == previous_u)

        backward_move = np.zeros(env.num_envs, dtype=np.int64)
        success = np.zeros(env.num_envs, dtype=bool)
        optimal_cost = np.empty(env.num_envs, dtype=np.float64)
        for i, graph in enumerate(env.graphs):
            optimal_cost[i] = graph.phi[graph.src]
            if valid[i]:
                backward_move[i] = (
                    graph.phi[destination[i]] > graph.phi[current_u[i]])
                success[i] = destination[i] == graph.tgt

        next_steps = env.steps + 1
        next_backward = env.backward + backward_move
        next_cost = env.path_cost + np.where(
            valid, weights[rows, actions], 0.0)
        done = success | (next_steps >= env.max_steps)

        observations, _, terminals, truncations, _ = env.step(actions)
        if not np.array_equal(done, terminals | truncations):
            raise RuntimeError('pre-step completion accounting diverged from env')

        for i in np.flatnonzero(done):
            if len(completed['success']) >= episodes:
                break
            n_steps = int(next_steps[i])
            is_success = bool(success[i])
            completed['success'].append(float(is_success))
            completed['length'].append(float(n_steps))
            completed['backward_rate'].append(
                float(next_backward[i] / max(1, n_steps)))
            completed['invalid_rate'].append(
                float(invalid_moves[i] / max(1, n_steps)))
            completed['two_cycle_rate'].append(
                float(two_cycles[i] / max(1, n_steps - 1)))
            completed['has_two_cycle'].append(float(two_cycles[i] > 0))
            if is_success:
                completed['opt_gap'].append(
                    float(next_cost[i] / max(optimal_cost[i], 1e-9)))

        previous_u[:] = current_u
        previous_u[done] = -1
        two_cycles[done] = 0
        invalid_moves[done] = 0
        if reset_done is not None:
            reset_done(done)

    return {
        'episodes': episodes,
        'success_rate': float(np.mean(completed['success'])),
        'episode_length': float(np.mean(completed['length'])),
        'opt_gap': (
            float(np.mean(completed['opt_gap']))
            if completed['opt_gap'] else float('nan')
        ),
        'backward_rate': float(np.mean(completed['backward_rate'])),
        'invalid_rate': float(np.mean(completed['invalid_rate'])),
        'two_cycle_rate': float(np.mean(completed['two_cycle_rate'])),
        'two_cycle_episode_rate': float(np.mean(completed['has_two_cycle'])),
    }
