"""Mac smoke run without pufferlib.

pufferlib==3.0.0 ships no macOS arm64 wheel (uv --only-binary finds none), so
this installs a minimal `pufferlib.PufferEnv` shim (buffer allocation only) and
evaluates the env with the scripted baselines plus a Dijkstra oracle that picks
the slot minimising Δ(u,v) = w - (φ(u) - φ(v)) (0 on shortest-path edges).
CPU-only, seconds to run.

    python scripts/mac_smoke.py --episodes 512 --n 8 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pufferlib  # noqa: F401
    SHIM = False
except ModuleNotFoundError:
    SHIM = True

    class PufferEnv:
        def __init__(self, buf=None):
            n = self.num_agents
            obs = self.single_observation_space
            self.observations = np.zeros((n, *obs.shape), dtype=obs.dtype)
            self.rewards = np.zeros(n, dtype=np.float32)
            self.terminals = np.zeros(n, dtype=bool)
            self.truncations = np.zeros(n, dtype=bool)
            self.masks = np.ones(n, dtype=bool)
            self.actions = np.zeros(n, dtype=np.int64)

    sys.modules['pufferlib'] = types.SimpleNamespace(PufferEnv=PufferEnv)

from dijkwalk.env import GraphWalk  # noqa: E402
from dijkwalk.evaluation import baseline_actions, evaluate_episodes  # noqa: E402


def oracle(_obs, env):
    slots = np.stack(env.slots)
    delta = np.stack(env.slot_w) - np.stack(env.slot_dphi)
    return np.where(slots >= 0, delta, np.inf).argmin(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--n', type=int, nargs=2, default=[8, 16])
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    ctrls = {
        'random-valid': baseline_actions('random-valid', rng),
        'greedy-weight': baseline_actions('greedy-weight', rng),
        'oracle-dphi': oracle,
    }
    print(f'pufferlib shim: {SHIM}  n=[{a.n[0]},{a.n[1]}]  episodes={a.episodes}')
    for name, fn in ctrls.items():
        env = GraphWalk(num_envs=a.num_envs, n_min=a.n[0], n_max=a.n[1],
                        n_max_final=a.n[1], curriculum=False, seed=a.seed)
        t = time.perf_counter()
        r = evaluate_episodes(env, fn, a.episodes, a.seed)
        r['seconds'] = round(time.perf_counter() - t, 2)
        print(name, json.dumps(r))


if __name__ == '__main__':
    main()
