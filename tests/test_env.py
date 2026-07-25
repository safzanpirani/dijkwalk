import numpy as np
import torch

from dijkwalk import graphgen
from dijkwalk.env import GraphWalk, K, F, OBS_DIM
from dijkwalk.policy import Policy, Transformer, MASK_VALUE


def make_env(**kw):
    kw.setdefault('num_envs', 16)
    kw.setdefault('seed', 7)
    kw.setdefault('curriculum', False)
    env = GraphWalk(**kw)
    env.reset(seed=kw['seed'])
    return env


def test_slots_match_adjacency():
    """Valid slots == true neighbors of u (or K cheapest if deg > K)."""
    env = make_env(num_envs=64)
    for _ in range(50):
        for i in range(env.num_envs):
            g, u = env.graphs[i], env.u[i]
            slots = env.slots[i]
            valid = slots[slots >= 0]
            adj = {v: w for v, w in g.adj[u]}
            assert len(valid) == min(len(adj), K)
            for k, v in enumerate(slots):
                if v < 0:
                    assert env.observations[i, k * F] == 0.0
                    continue
                assert v in adj, f'slot {k} node {v} not a neighbor of {u}'
                assert env.slot_w[i][k] == adj[v]
                assert env.observations[i, k * F] == 1.0
                assert np.isclose(env.observations[i, k * F + 1], adj[v] / env.w_max)
        acts = random_valid_actions(env)
        env.step(acts)


def random_valid_actions(env, rng=np.random.default_rng(0)):
    acts = np.zeros(env.num_envs, dtype=np.int64)
    for i in range(env.num_envs):
        valid = np.flatnonzero(env.slots[i] >= 0)
        acts[i] = rng.choice(valid)
    return acts


def test_graded_reward_is_minus_delta():
    """With beta=1, step_cost=0: r == -Delta(u,v)/w_max exactly (Ng shaping)."""
    env = make_env(num_envs=32, beta=1.0, step_cost=0.0, r_goal=0.0)
    rng = np.random.default_rng(1)
    for _ in range(100):
        pre = [(env.graphs[i], int(env.u[i])) for i in range(env.num_envs)]
        acts = random_valid_actions(env, rng)
        moves = [int(env.slots[i][acts[i]]) for i in range(env.num_envs)]
        _, rewards, _, _, _ = env.step(acts)
        for i in range(env.num_envs):
            g, u = pre[i]
            delta = graphgen.step_deltas(g, u)[moves[i]]
            assert np.isclose(rewards[i], -delta / env.w_max, atol=1e-6), \
                f'r={rewards[i]} vs -delta/wmax={-delta / env.w_max}'


def test_optimal_path_telescopes():
    """Walking the optimal path: every step reward is 0 (beta=1, no costs),
    and the episode terminates with success."""
    env = make_env(num_envs=4, beta=1.0, step_cost=0.0, r_goal=10.0)
    for i in range(env.num_envs):
        # other envs get random-stepped while we walk env i, so derive the
        # optimal path from env i's *current* graph and position
        g = env.graphs[i]
        path = graphgen.optimal_path(g.adj, g.phi, int(env.u[i]), g.tgt)
        for u, v in zip(path, path[1:]):
            assert env.u[i] == u
            slot = int(np.flatnonzero(env.slots[i] == v)[0])
            acts = random_valid_actions(env)
            acts[i] = slot
            # freeze other envs' randomness effects by checking only env i
            _, rewards, terms, _, _ = env.step(acts)
            if v == g.tgt:
                assert terms[i] and np.isclose(rewards[i], 10.0, atol=1e-6)
                break
            assert np.isclose(rewards[i], 0.0, atol=1e-6)


def test_truncation_and_invalid_action():
    env = make_env(num_envs=2, n_min=8, n_max=8)
    g = env.graphs[0]
    deg = len(g.adj[g.src])
    if deg < K:  # pick a padded slot: must be a penalized no-op
        acts = np.array([deg, 0])
        u_before = env.u[0]
        _, rewards, terms, truncs, _ = env.step(acts)
        assert env.u[0] == u_before and not terms[0]
        assert rewards[0] < -env.penalty + 1e-6
    # run until truncation fires somewhere
    fired = False
    for _ in range(10 * 8 * 4):
        _, _, terms, truncs, _ = env.step(random_valid_actions(env))
        if truncs.any() or terms.any():
            fired = True
    assert fired


def test_policy_masks_invalid_slots():
    env = make_env(num_envs=8)
    obs = torch.from_numpy(env.observations.copy())
    for net in (Policy(env), Transformer(env)):
        logits, values = net.forward_eval(obs)
        assert logits.shape == (8, K) and values.shape == (8, 1)
        invalid = torch.from_numpy(
            np.stack([env.slots[i] < 0 for i in range(8)]))
        assert (logits[invalid] <= MASK_VALUE / 2).all(), \
            f'{type(net).__name__} leaked logits on invalid slots'
        assert (logits[~invalid] > MASK_VALUE / 2).all()


def test_observation_bounds():
    env = make_env(num_envs=32)
    rng = np.random.default_rng(3)
    for _ in range(200):
        env.step(random_valid_actions(env, rng))
        assert (env.observations >= 0).all() and (env.observations <= 1).all()


def test_seed_controls_graph_generation():
    first = make_env(num_envs=4, seed=123)
    second = make_env(num_envs=4, seed=123)
    different = make_env(num_envs=4, seed=124)

    def signatures(env):
        return [
            (g.n, g.edges, g.src, g.tgt, g.path)
            for g in env.graphs
        ]

    assert signatures(first) == signatures(second)
    assert signatures(first) != signatures(different)
