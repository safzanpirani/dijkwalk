"""Evaluate classical graph-walk baselines across graph sizes and seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dijkwalk.env import GraphWalk
from dijkwalk.evaluation import baseline_actions, evaluate_episodes

METRICS = (
    'success_rate',
    'episode_length',
    'opt_gap',
    'backward_rate',
    'invalid_rate',
    'two_cycle_rate',
    'two_cycle_episode_rate',
)


def run_job(job):
    method, n, seed, episodes, num_envs, max_steps_factor = job
    env = GraphWalk(
        num_envs=num_envs,
        n_min=n,
        n_max=n,
        curriculum=False,
        max_steps_factor=max_steps_factor,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1_000_003)
    result = evaluate_episodes(
        env,
        baseline_actions(method, rng),
        episodes=episodes,
        seed=seed,
    )
    result.update(
        method=method,
        n=n,
        seed=seed,
        max_steps_factor=max_steps_factor,
    )
    return result


def aggregate(results):
    groups = {}
    for result in results:
        key = (
            result['method'],
            result['n'],
            result['max_steps_factor'],
        )
        groups.setdefault(key, []).append(result)

    summary = []
    for (method, n, max_steps_factor), runs in sorted(groups.items()):
        row = {
            'method': method,
            'n': n,
            'max_steps_factor': max_steps_factor,
            'seeds': [run['seed'] for run in runs],
        }
        for metric in METRICS:
            values = np.asarray([run[metric] for run in runs], dtype=float)
            row[metric] = {
                'mean': float(np.nanmean(values)),
                'std': float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['random-valid', 'greedy-weight'],
    )
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[32, 48, 96, 192, 384, 768],
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--max-steps-factor', type=float, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--out',
        default='paper-results/q4-classical-baselines.json',
    )
    args = parser.parse_args()

    jobs = [
        (
            method,
            n,
            seed,
            args.episodes,
            args.num_envs,
            args.max_steps_factor,
        )
        for method in args.methods
        for n in args.sizes
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_job, jobs))

    payload = {
        'config': {
            'methods': args.methods,
            'sizes': args.sizes,
            'seeds': args.seeds,
            'episodes_per_seed': args.episodes,
            'num_envs': args.num_envs,
            'max_steps_factor': args.max_steps_factor,
        },
        'runs': results,
        'summary': aggregate(results),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(f'wrote {args.out} ({len(results)} runs)')


if __name__ == '__main__':
    main()
