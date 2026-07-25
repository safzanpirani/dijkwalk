"""Compare the vectorized environment with the scalar Q3 implementation."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch  # preloads dylibs for pufferlib._C
import pufferlib.vector
from pufferlib import pufferl

from dijkwalk.env import GraphWalk
from dijkwalk.policy import Policy, Recurrent
from scripts.run_experiments import base_config, seed_everything


def load_scalar_class(git_ref):
    source = subprocess.check_output(
        ['git', 'show', f'{git_ref}:dijkwalk/env.py'],
        text=True,
    )
    source = source.replace(
        'from . import graphgen',
        'from dijkwalk import graphgen',
    )
    module = types.ModuleType('scalar_env')
    exec(compile(source, 'scalar_env.py', 'exec'), module.__dict__)
    return module.GraphWalk


def valid_actions(env, rng):
    slots = env.slots if isinstance(env.slots, np.ndarray) else np.stack(env.slots)
    counts = (slots >= 0).sum(axis=1)
    return (rng.random(env.num_envs) * counts).astype(np.int64)


def environment_sps(env_cls, seed, steps, prevent_resets):
    max_steps_factor = 100_000 if prevent_resets else 4
    env = env_cls(
        num_envs=1024,
        n_min=32,
        n_max=64,
        curriculum=False,
        max_steps_factor=max_steps_factor,
        report_interval=32,
        seed=seed,
    )
    env.reset(seed=seed)
    if prevent_resets:
        for graph in env.graphs:
            graph.tgt = -1
        if hasattr(env, 'tgt'):
            env.tgt.fill(-1)
        env.max_steps.fill(1_000_000_000)

    rng = np.random.default_rng(seed + 10_000)
    for _ in range(50):
        env.step(valid_actions(env, rng))
    start = time.perf_counter()
    for _ in range(steps):
        env.step(valid_actions(env, rng))
    elapsed = time.perf_counter() - start
    return env.num_envs * steps / elapsed


def ppo_sps(env_cls, seed, steps):
    seed_everything(seed)
    config = base_config(steps, 1024, 'cpu', seed)
    config['env'].update(
        n_min=32,
        n_max=64,
        n_max_final=64,
        curriculum=False,
        max_steps_factor=4,
        report_interval=32,
    )
    config['train']['use_rnn'] = True
    with contextlib.redirect_stdout(io.StringIO()):
        vecenv = pufferlib.vector.make(
            lambda **kwargs: env_cls(**kwargs),
            env_kwargs=config['env'],
            backend='Serial',
            num_envs=1,
            seed=seed,
        )
        policy = Recurrent(
            vecenv.driver_env,
            Policy(vecenv.driver_env, 128),
            128,
            128,
        )
        trainer = pufferl.PuffeRL(config['train'], vecenv, policy)
        start = time.perf_counter()
        while trainer.global_step < steps:
            trainer.evaluate()
            trainer.train()
        elapsed = time.perf_counter() - start
        completed = trainer.global_step
        trainer.close()
    return completed / elapsed


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        'values': values.tolist(),
        'mean': float(values.mean()),
        'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def comparison(scalar_values, vector_values):
    scalar = summarize(scalar_values)
    vector = summarize(vector_values)
    return {
        'scalar': scalar,
        'vectorized': vector,
        'speedup': vector['mean'] / scalar['mean'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalar-ref', default='51e3715')
    parser.add_argument('--seeds', nargs='+', type=int, default=[801, 802, 803])
    parser.add_argument('--environment-steps', type=int, default=1000)
    parser.add_argument('--ppo-steps', type=int, default=500_000)
    parser.add_argument(
        '--out',
        default='paper-results/vectorization-benchmark.json',
    )
    args = parser.parse_args()

    scalar_class = load_scalar_class(args.scalar_ref)
    scalar_end_to_end = [
        environment_sps(
            scalar_class, seed, args.environment_steps, prevent_resets=False)
        for seed in args.seeds
    ]
    vector_end_to_end = [
        environment_sps(
            GraphWalk, seed, args.environment_steps, prevent_resets=False)
        for seed in args.seeds
    ]
    scalar_kernel = [
        environment_sps(
            scalar_class, seed + 10, args.environment_steps,
            prevent_resets=True)
        for seed in args.seeds
    ]
    vector_kernel = [
        environment_sps(
            GraphWalk, seed + 10, args.environment_steps,
            prevent_resets=True)
        for seed in args.seeds
    ]
    scalar_ppo = ppo_sps(scalar_class, seed=823, steps=args.ppo_steps)
    vector_ppo = ppo_sps(GraphWalk, seed=823, steps=args.ppo_steps)

    result = {
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
            'python': platform.python_version(),
            'torch': torch.__version__,
        },
        'config': {
            'scalar_ref': args.scalar_ref,
            'seeds': args.seeds,
            'num_envs': 1024,
            'n_min': 32,
            'n_max': 64,
            'max_steps_factor': 4,
            'environment_steps': args.environment_steps,
            'ppo_steps': args.ppo_steps,
            'thread_cap': 1,
        },
        'environment_with_resets_sps': comparison(
            scalar_end_to_end, vector_end_to_end),
        'transition_kernel_no_resets_sps': comparison(
            scalar_kernel, vector_kernel),
        'lstm_muon_ppo_sps': comparison([scalar_ppo], [vector_ppo]),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
